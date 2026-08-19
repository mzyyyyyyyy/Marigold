import collections
import glob
import json
import os
import pickle
import random
import sys
import time
import multiprocessing
import concurrent.futures
import re
from pathlib import Path
import numpy as np
import rasterio
import rasterio.errors
from rasterio.windows import Window
from skimage.transform import resize
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2



# Bounded LRU, not a plain dict: with num_workers=6 and 630+ training steps
# touching many distinct source rasters over time, an unbounded cache means
# every worker process ends up holding every source file it has ever opened
# open forever (each with its own GDAL block cache on top) — suspected
# root cause of the host-RAM OOMs that consistently killed runs around
# step 630 (see 2026-08-18 investigation: per-rank RSS measured via psutil
# stayed flat/modest, but sacct MaxRSS for the whole cgroup — main process +
# its 6 DataLoader workers — hit 90-100+GB, meaning the growth was in the
# uninstrumented worker processes, not the main one). Capping the number of
# concurrently open datasets per process bounds this regardless of how many
# distinct files get touched over an arbitrarily long run.
_RASTERIO_CACHE_MAX = 8
_rasterio_cache: "collections.OrderedDict[str, rasterio.io.DatasetReader]" = collections.OrderedDict()


def _log_err(msg: str) -> None:
    """Write straight to stderr, tagged with this process's SLURM rank.

    Deliberately bypasses the `logging` module: config_logging() (in
    train_sr_fm_refiner.py) only attaches handlers on rank0, so
    logging.warning/error would land in logging.log + train_<job>.out on
    rank0 but silently fall back to Python's unformatted, rank-less
    last-resort stderr handler on every other rank. Writing directly to
    stderr, always, from every rank, guarantees these all land in the one
    place SLURM already collects every rank's stderr — train_<job>.err —
    with a consistent, rank-tagged format regardless of which rank hits it.
    """
    rank = os.environ.get("SLURM_PROCID", "?")
    print(f"[ps_lazydataset][rank {rank}] {msg}", file=sys.stderr, flush=True)


def _tile_id_to_padded(tile_id: str, width: int = 5) -> str:
    """'6,55' -> '00006_00055' — the zero-padded underscore convention used by
    PlanetScope HR / CHM output filenames (vs. the comma-separated 'R,C' used
    by Landsat filenames)."""
    row, col = tile_id.split(',')
    return f"{int(row):0{width}d}_{int(col):0{width}d}"


class MinMaxScale(A.ImageOnlyTransform):
    def __init__(self, min_vals, max_vals, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.min_vals = np.array(min_vals, dtype=np.float32)
        self.max_vals = np.array(max_vals, dtype=np.float32)

    def apply(self, image, **params):
        # image shape: (H, W, C)
        image = (image - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        image = np.clip(image, 0, 1)
        return image


class ConvertDtype(A.ImageOnlyTransform):
    """Convert float64 to float32 to avoid albumentations dtype issues"""
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.astype(np.float32)


class LazyPatchDataset(Dataset):
    def __init__(self, config):
        # 读取多模态文件地址，输出patch信息成pkl文件；如果pkl文件已经存在，则直接读取
        self.input_dir = config['input_dir']
        self.input_dir_hr = config['input_dir_hr']
        self.output_dir = config['output_dir']
        self.selected_percentile = config['selected_percentile']
        self.file_type_input = config['file_type_input']
        self.file_type_output = config['file_type_output']
        self.patch_sizes = config['patch_sizes']
        self.patch_coord_path = config['patch_coord_path']
        self.patches_per_tile = config['num_patches_per_tile']
        self.tile_emphasis = config['tile_emphasis']
        self.selected_bands = config['selected_bands']
        self.selected_bands_hr = config['selected_bands_hr']
        self.correlation_cleaning = config['correlation_cleaning']
        self.target_range_edges = config['target_range_edges']
        self.num_patches_per_target_range = config['num_patches_per_target_range']
        self.year = config['year']

        self.disp_name = "ps_mm_dataset"
        self.filename_ls_path = None
        self.patch_coords = self._compute_coords()

        self.batch_size = config['batch_size']
        self.mode = config['mode']
        self.use_input_minmax = config['use_input_minmax']
        self.use_input_norm = config['use_input_norm']
        self.input_stats_file = config['input_stats_file']
        self.use_input_hr_minmax = config['use_input_hr_minmax']
        self.use_input_hr_norm = config['use_input_hr_norm']
        self.input_hr_stats_file = config['input_hr_stats_file']
        self.use_target_minmax = config['use_target_minmax']
        self.use_target_norm = config['use_target_norm']
        self.target_stats_file = config['target_stats_file']
        self.unit_scale_ratio = config['unit_scale_ratio']
        self.scale_input_to_neg1_1 = config.get('scale_input_to_neg1_1', False)
        self.scale_input_hr_to_neg1_1 = config.get('scale_input_hr_to_neg1_1', False)
        self.scale_target_to_neg1_1 = config.get('scale_target_to_neg1_1', False)

        if self.use_input_minmax or self.use_input_norm:
            self.input_stats = self._read_stats(self.input_stats_file)
        if self.use_input_hr_minmax or self.use_input_hr_norm:
            self.input_hr_stats = self._read_stats(self.input_hr_stats_file)
        if self.use_target_minmax or self.use_target_norm:
            self.target_stats = self._read_stats(self.target_stats_file)


        # 根据 patch_size 把 patch_coords 分成不同的组，方便后续根据不同 patch_size 采样
        self.patches_by_size = {}
        for patch_size in self.patch_sizes:
            # Filter patches by size from patch_coordinates
            size_patches = [
                patch for patch in self.patch_coords
                if patch['output_patch_height'] == patch_size[0] and patch['output_patch_width'] == patch_size[1]
            ]
            self.patches_by_size[patch_size] = size_patches
        
        print(f"Generated patches for each size:")
        for size, patches in self.patches_by_size.items():
            print(f"  {size}: {len(patches)} patches")

        
        # 构建输入输出的 transform pipeline
        self.transform_input, self.transform_input_hr, self.transform_output = self._transform()

    def _read_stats(self, stats_file):
        with open(stats_file, "r") as f:
            all_stats = json.load(f)
            try:
                stats = all_stats[str(self.year)]
            except:
                print(f'Can not load input statistics for year {self.year}')
        return stats
    
    def _compute_coords(self):
        if not os.path.exists(self.patch_coord_path):
            raise ValueError('Please specify a correct path (file or dir) to save the coordinates for training data preparation')

        if os.path.isdir(self.patch_coord_path):
            tile_ids = self._gather_tile_ids()
            coords = self._generate_patch_coordinates_parallel(tile_ids)
            
            try:
                from datetime import datetime
                save_patch_path = os.path.join(self.patch_coord_path, 'save_' + datetime.now().strftime("%Y%m%d") + '.pkl')
                with open(save_patch_path, "wb") as f:
                    pickle.dump(coords, f)
            except:
                return False
            
            return coords  
        
        elif os.path.isfile(self.patch_coord_path):
            
            print('------------------------------------')
            print('Loading preprocessed patch coordinates from file path: ', self.patch_coord_path)
            with open(self.patch_coord_path, "rb") as f:
                coords = pickle.load(f)
                self.patch_coords = coords
                self._check_coord_path_match_config_path()
                self._check_coord_scale_match_data_scale()
                return coords
        else:
            raise ValueError('Coord path should be either dir or file')

    def _gather_tile_ids(self):
        """Gather all tile IDs from the input directory,
        gather all tile ids from the output directory,
        take the intersection of the two lists to form the tile_ids"""
        
        all_input_files = glob.glob(f'{self.input_dir}/*/*{self.selected_percentile[0]}*.{self.file_type_input}') # for landsat
        all_input_hr_files = glob.glob(f'{self.input_dir_hr}/*.{self.file_type_input}') # for planetscope
        all_output_files = glob.glob(f'{self.output_dir}/*.{self.file_type_output}') # for chm dataset, both landsat and planetscope
        print(f"Found {len(all_input_files)} LR input files, {len(all_input_hr_files)} HR input files, {len(all_output_files)} output files")

        tiles_input = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0] for file in all_input_files] # for landsat

        tiles_input_hr = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0:2] for file in all_input_hr_files] # for planetscope
        tiles_input_hr = [(int(t[0]), int(t[1])) for t in tiles_input_hr]
        tiles_input_hr = [f"{x},{y}" for x, y in tiles_input_hr]

        tiles_output = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0:2] for file in all_output_files] # for chm, both ls and ps
        tiles_output = [(int(t[0]), int(t[1])) for t in tiles_output]
        tiles_output = [f"{x},{y}" for x, y in tiles_output]

        tile_ids = list(set(tiles_input) & set(tiles_input_hr) & set(tiles_output))
        print(f"After filtering [existing both input and output], found {len(tile_ids)} tiles for training and testing!")
        return tile_ids

    def _generate_patch_coordinates_parallel(self, tile_ids):
        manager = multiprocessing.Manager()
        progress_counter = manager.Value('i', 0)

        total_patches = len(tile_ids) * self.patches_per_tile * len(self.patch_sizes)  # rough estimate
        all_patch_coords = []
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.7))

        with tqdm(total=total_patches, desc="Processing patches", dynamic_ncols=True) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        LazyPatchDataset._process_tile,
                        tile_id,
                        self.input_dir,
                        self.input_dir_hr,
                        self.output_dir,
                        self.selected_percentile,
                        self.file_type_input,
                        self.patch_sizes,
                        self.patches_per_tile,
                        progress_counter,
                        self.tile_emphasis,
                        self.selected_bands,
                        self.correlation_cleaning,
                        self.target_range_edges,
                        self.num_patches_per_target_range
                    )
                    for tile_id in tile_ids
                ]

                while any(future.running() for future in futures):
                    # Update the bar based on shared counter
                    pbar.n = progress_counter.value
                    pbar.refresh()

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        all_patch_coords.extend(result)


        if not all_patch_coords:
            raise RuntimeError(
                "No patch coordinates were generated. "
                "Check that input_dir, input_dir_hr, and output_dir contain matching tiles, "
                "and that tile processing did not fail silently (see error messages above)."
            )

        # extract scale factor
        self.in_out_scale_factor = all_patch_coords[0]['scale_factor_x']
        self.out_in_scale_factor = 1.0 / self.in_out_scale_factor
        print('very rough estimate: data left after filtering: ', len(all_patch_coords)/total_patches)
        return all_patch_coords

    @staticmethod
    def _process_tile(tile_id, input_dir, input_dir_hr, output_dir, selected_percentile, file_type, patch_sizes, patches_per_tile,
                     progress_counter, tile_emphasis, select_bands,
                    correlation_cleaning, target_range_edges, num_patches_per_target_range):
        """
        Modified to increment progress counter for every patch processed.
        """
        patch_coords = []
        if len(selected_percentile) > 1: # more than one perc
            indc = [14, 17]
        elif len(selected_percentile) == 1:
            indc = [0, 3]

        # tile_id_input = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}" # for planetscope and chm dataset
        # tile_id_output = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}" 

        tile_id_output = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}" # for landsat chm dataset. 


        try:

            input_path = glob.glob(f'{input_dir}/*/*{tile_id}*{selected_percentile[0]}*.{file_type}')[0]
            output_path = glob.glob(f'{output_dir}/*{tile_id_output}*.tif')[0]
            input_path_hr = glob.glob(f'{input_dir_hr}/*{tile_id_output}*.{file_type}')[0]

            with rasterio.open(input_path) as src:
                input_height, input_width = src.height, src.width
            with rasterio.open(input_path_hr) as src:
                input_height_hr, input_width_hr = src.height, src.width
            with rasterio.open(output_path) as src:
                output_height, output_width = src.height, src.width

            scale_factor_x = input_width / output_width # for landsat and ps
            scale_factor_y = input_height / output_height



            for patch_height, patch_width in patch_sizes:
                max_y_output = output_height - patch_height
                max_x_output = output_width - patch_width

                if max_y_output < 0 or max_x_output < 0:
                    continue
                if tile_id not in tile_emphasis:
                    n_tiles = patches_per_tile
                else:
                    n_tiles = 10*patches_per_tile

                # start of the cropping
                no_attempt = 0
                quota = {i: 0 for i in range(len(target_range_edges)-1)} # count in each group
                # print(f"max attempts is {n_tiles}")
                while no_attempt < n_tiles:
                    no_attempt += 1
            
                    y_output = random.randint(0, max_y_output)
                    x_output = random.randint(0, max_x_output)

                    y_input_hr = y_output
                    x_input_hr = x_output

                    y_input = round(y_output * scale_factor_y)
                    x_input = round(x_output * scale_factor_x)

                    input_patch_height = round(patch_height * scale_factor_y)
                    input_patch_width = round(patch_width * scale_factor_x)

                    # 由于 target 是由 hr 计算出来的，所以 input_hr 与 target 本来就是对齐的
                    input_patch_height_hr = patch_height
                    input_patch_width_hr = patch_width

                    if (y_input + input_patch_height > input_height or 
                        x_input + input_patch_width > input_width):
                        continue

                    # 至此，实现从遥感影像上随机裁剪出一块 patch 的坐标了，
                    # 接下来要判断这块 patch 是否满足条件（比如不是全0，或者在某个范围内），
                    # 如果满足条件才加入到训练数据中
                    output_patch = LazyPatchDataset._load_output_patch(
                        output_path, y_output, x_output, patch_height, patch_width
                    )

                    # Using a very low threshold will filter out too many patches with low values (no data, where it;s supposed to learn 0)
                    if np.sum(output_patch == 0) / output_patch.size > 0.95:
                        progress_counter.value += 1
                        continue
                    # ipdb.set_trace()

                    proxyi = np.percentile(output_patch, 75) # use this to define group of the patch
                    group_idx = np.digitize([proxyi], target_range_edges[1:-1])[0]
                    # ipdb.set_trace()
                    if quota[group_idx] > num_patches_per_target_range:
                        progress_counter.value += 1
                        continue
                    quota[group_idx] += 1
                    
                    # continue cropping
                    if correlation_cleaning:
                        input_patch = LazyPatchDataset._load_multiband_patch(
                            input_path, y_input, x_input, input_patch_height, input_patch_width, select_bands
                        )
                        redb = input_patch[..., indc[0]].astype(float)
                        nirb = input_patch[..., indc[1]].astype(float)
                        ndvi = (nirb - redb) / (nirb + redb + 1e-10)
                        downsampled_ndvi = resize(ndvi, (output_patch.shape[0], output_patch.shape[1]),
                                                    order=0, anti_aliasing=True, preserve_range=True)

                        rho, _ = spearmanr(downsampled_ndvi.flatten(), output_patch.flatten())
                        # rho = np.corrcoef(downsampled_ndvi.flatten(), output_patch.flatten())[0, 1]
                        if rho < 0:
                            progress_counter.value += 1
                            quota[group_idx] -= 1
                            continue



                    patch_coords.append({
                        'tile_id': tile_id,
                        'input_path': input_path,
                        'output_path': output_path,
                        'input_crop_y': y_input,
                        'input_crop_x': x_input,
                        'input_patch_height': input_patch_height,
                        'input_patch_width': input_patch_width,
                        'input_path_hr': input_path_hr,
                        'input_crop_y_hr': y_input_hr,
                        'input_crop_x_hr': x_input_hr,
                        'input_patch_height_hr': input_patch_height_hr,
                        'input_patch_width_hr': input_patch_width_hr,
                        'output_crop_y': y_output,
                        'output_crop_x': x_output,
                        'output_patch_height': patch_height,
                        'output_patch_width': patch_width,
                        'scale_factor_x': scale_factor_x,
                        'scale_factor_y': scale_factor_y
                    })


                    progress_counter.value += 1 

        except Exception as e:
            print(f"Error processing tile {tile_id}: {e}")

        return patch_coords

    @staticmethod
    def _load_multiband_patch(filepath: str, y: int, x: int, height: int, width: int, select_bands: list,
                             ignore_percentile = True) -> np.ndarray:
        """Load a patch from a multi-band image with selected bands
            return_tile_id: used to build geo encoders

        """

        if ignore_percentile:
            window = Window(x, y, width, height)
            image = LazyPatchDataset._read_window_with_retry(filepath, window)

            # Select only the specified bands
            max_band_idx = max(select_bands)
            if max_band_idx >= image.shape[0]:
                available_bands = list(range(min(len(select_bands), image.shape[0])))
                selected_image = image[available_bands]
            else:
                selected_image = image[select_bands]

            images = np.transpose(selected_image, (1, 2, 0))

        else:
            raise ValueError("Multiple percentiles are not supported. Set ignore_percentile=True.")

        return images

    @staticmethod
    def _load_output_patch(filepath: str, y: int, x: int, height: int, width: int) -> np.ndarray:
        """Load a patch from output image"""
        if filepath.endswith('.tif'):
            window = Window(x, y, width, height)
            image = LazyPatchDataset._read_window_with_retry(filepath, window)
            if len(image.shape) == 3:
                image = image[0]

            # Check for invalid values before any cleaning
            inf_count = np.sum(np.isinf(image))
            nan_count = np.sum(np.isnan(image))
            if inf_count > 0 or nan_count > 0:
                print(f"Raw data has {inf_count} inf and {nan_count} NaN values")

            return image
        else:
            raise ValueError(f"Unsupported file format: {filepath}. Only .tif files are supported.")

    @staticmethod
    def _open_rasterio(filepath: str):
        """Return a cached rasterio dataset for this worker process.

        Bounded LRU (max _RASTERIO_CACHE_MAX open datasets): touching this
        file again just marks it as most-recently-used; opening a new file
        beyond the cap closes whichever cached dataset was least recently
        used, so a long run touching many distinct source files can't leave
        every one of them open (and cached in GDAL's per-dataset block
        cache) forever.
        """
        if filepath in _rasterio_cache:
            _rasterio_cache.move_to_end(filepath)
            return _rasterio_cache[filepath]
        if len(_rasterio_cache) >= _RASTERIO_CACHE_MAX:
            _, evicted = _rasterio_cache.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        src = rasterio.open(filepath)
        _rasterio_cache[filepath] = src
        return src

    @staticmethod
    def _read_window_with_retry(filepath: str, window: Window, max_retries: int = 3,
                                 backoff_seconds: float = 0.5) -> np.ndarray:
        """Read a window, retrying on RasterioIOError.

        Verified (see 2026-08-17 investigation) that the source files behind
        these errors are NOT corrupted — a full sequential single-process
        read of every band/block succeeds cleanly. The failures only show up
        under heavy DDP training load, where the deterministic sampling
        order sends many ranks'/workers' DataLoader reads at the same large
        source file in a short burst, spiking per-file concurrent-read
        pressure and occasionally producing a transient torn/short read.
        Retrying (with a fresh dataset handle, in case the cached handle
        itself — not just the read — is what's wedged) rides out that
        transient spike instead of killing the whole 64-GPU job over it.
        """
        last_err = None
        for attempt in range(max_retries):
            try:
                src = LazyPatchDataset._open_rasterio(filepath)
                return src.read(window=window)
            except rasterio.errors.RasterioIOError as e:
                last_err = e
                _log_err(
                    f"Transient read failure on {filepath} "
                    f"(window={window}), attempt {attempt + 1}/{max_retries}: {e}"
                )
                stale = _rasterio_cache.pop(filepath, None)
                if stale is not None:
                    try:
                        stale.close()
                    except Exception:
                        pass
                if attempt < max_retries - 1:
                    time.sleep(backoff_seconds * (attempt + 1))
        _log_err(
            f"Giving up on {filepath} (window={window}) after "
            f"{max_retries} retries: {last_err}"
        )
        raise last_err

    def _check_coord_path_match_config_path(self):
        """If coord file is given, double check whether the contained input, input_hr, and output file path match with those given in the
        new config file, otherwise replace the saved path with the new path.

        Useful when coords are saved, but input, input_hr, or output files are changed (without a need to rerun coordinates sampling)
        """
        print('Double checking whether pkl saved input/output paths match with config file paths')
        random_entry = random.choice(self.patch_coords)
        # ipdb.set_trace()
        try:
            Path(random_entry.get('input_path')).resolve().relative_to(Path(self.input_dir).resolve())
            input_match = True
        except:
            input_match = False
        try:
            Path(random_entry.get('input_path_hr')).resolve().relative_to(Path(self.input_dir_hr).resolve())
            input_hr_match = True
        except:
            input_hr_match = False
        try:
            Path(random_entry.get('output_path')).resolve().relative_to(Path(self.output_dir).resolve())
            output_match = True
        except:
            output_match = False
        if input_match and input_hr_match and output_match:
            print("A random entry's paths match the config directories.")
            return

        if not input_match:
            print(f"random entry's input path is {random_entry.get('input_path')}")
            print(f"config input dir is {self.input_dir}")
            print('*' * 10 + 'Rewriting coord path input dirs to config paths')
            pattern = re.compile(r'(\d+,\d+)')
            for element in tqdm(self.patch_coords):
                filename = os.path.basename(element['input_path'])
                coordinate = pattern.search(filename).group(1)
                search_pattern = os.path.join(self.input_dir, f"**/*{coordinate}*.{self.file_type_input}")
                found_file = glob.glob(search_pattern, recursive=True)[0]
                assert os.path.exists(found_file)
                element['input_path'] = found_file

        if not input_hr_match:
            print(f"random entry's input_hr path is {random_entry.get('input_path_hr')}")
            print(f"config input_hr dir is {self.input_dir_hr}")
            print('*' * 10 + 'Rewriting coord path input_hr dirs to config paths')
            # PlanetScope HR filenames encode the tile as zero-padded
            # "RRRRR_CCCCC" (e.g. tile_id "6,55" -> "00006_00055"), not the
            # comma-separated "R,C" used by Landsat filenames — so derive the
            # search pattern from tile_id instead of regexing the filename.
            for element in tqdm(self.patch_coords):
                coordinate = _tile_id_to_padded(element['tile_id'])
                search_pattern = os.path.join(self.input_dir_hr, f"*{coordinate}*.{self.file_type_input}")
                found_file = glob.glob(search_pattern)[0]
                assert os.path.exists(found_file)
                element['input_path_hr'] = found_file

        if not output_match:
            print(f"random entry's output path is {random_entry.get('output_path')}")
            print(f"config output dir is {self.output_dir}")
            print('*' * 10 + 'Rewriting coord path output dirs to config paths')
            # Same zero-padded "RRRRR_CCCCC" convention as the HR files above.
            for element in tqdm(self.patch_coords):
                coordinate = _tile_id_to_padded(element['tile_id'])
                search_pattern = os.path.join(self.output_dir, f"*{coordinate}*.{self.file_type_output}")
                found_file = glob.glob(search_pattern)[0]
                assert os.path.exists(found_file)
                element['output_path'] = found_file

        # Persist the corrected paths to a *new* pkl (never overwrite the
        # original — it may still be the one other in-flight configs/jobs
        # load): this glob-per-patch rewrite is expensive (one glob.glob()
        # per patch, per mismatched field), and every process that
        # constructs this dataset against the same stale pkl would otherwise
        # redo it independently. In a multi-rank DDP run that means dozens
        # of processes hammering the same shared filesystem with redundant
        # glob storms at once. Pointing patch_coord_path at the rewritten
        # pkl makes the check above pass immediately next time, on every
        # rank, with no glob calls at all. Written atomically (temp file +
        # rename) so a run killed mid-write can't corrupt a cache file other
        # processes may be reading concurrently.
        base, ext = os.path.splitext(self.patch_coord_path)
        resolved_path = f"{base}_resolved{ext}"
        print('*' * 10 + f'Saving corrected paths to {resolved_path} '
              f'(original {self.patch_coord_path} left untouched — '
              f'point patch_coord_path at the new file to reuse this fix)')
        tmp_path = f"{resolved_path}.tmp{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.patch_coords, f)
        os.replace(tmp_path, resolved_path)
        # ipdb.set_trace()

    def _check_coord_scale_match_data_scale(self):
        # check whether input/output spatial scale match, otherwise update scale!
        random_entry = random.choice(self.patch_coords)
        with rasterio.open(random_entry.get('input_path')) as src:
            input_height, input_width = src.height, src.width
        with rasterio.open(random_entry.get('output_path')) as src:
            output_height, output_width = src.height, src.width

        scale_factor = input_width / output_width
        out_in_scale_factor = output_width / input_width
        self.in_out_scale_factor = scale_factor
        self.out_in_scale_factor = out_in_scale_factor
        print(f'Confirmed: Input/Output dimension ratio is {self.in_out_scale_factor:.3f}')

    def _transform(self):
        # 根据 config 里的数据预处理、增强选项，构建输入输出的 transform pipeline
        # list of transforms for input data，这个顺序是合理的。
        transform_list_input = []
        if self.use_input_minmax:
            p1  = [self.input_stats["p1"][b]  for b in self.selected_bands]
            p99 = [self.input_stats["p99"][b] for b in self.selected_bands]
            min_max = MinMaxScale(p1, p99)
            transform_list_input.append(min_max)

        if self.mode == 'train':
            transform_list_input.extend([A.RandomBrightnessContrast(p=0.5),
                                        A.RandomGamma(p=0.5)])

        if self.use_input_norm:
            norm = A.Normalize(mean=self.input_stats['mean'], std=self.input_stats['std'])
            print(f"After min-max scaling, using patch-wise normalization for {len(self.selected_bands)} channels")
            print('=' * 100)
            # norm = PatchwiseNormalize()
            print('Using pre-calculated global mean std')
            transform_list_input.append(norm)
            
        if self.scale_input_to_neg1_1:
            transform_list_input.append(A.Lambda(image=lambda x, **kwargs: x * 2.0 - 1.0))
        transform_list_input.extend([ConvertDtype(),  # Convert float64 to float32 first
                                    ToTensorV2()])
        transform_list_input_compose = A.Compose(transform_list_input, is_check_shapes=False)

        # list of transforms for input_hr data
        transform_list_input_hr = []
        if self.use_input_hr_minmax:
            p1_hr  = [self.input_hr_stats["p1"][b]  for b in self.selected_bands_hr]
            p99_hr = [self.input_hr_stats["p99"][b] for b in self.selected_bands_hr]
            min_max_hr = MinMaxScale(p1_hr, p99_hr)
            transform_list_input_hr.append(min_max_hr)

        if self.mode == 'train':
            transform_list_input_hr.extend([A.RandomBrightnessContrast(p=0.5),
                                        A.RandomGamma(p=0.5)])

        if self.use_input_hr_norm:
            norm_hr = A.Normalize(mean=self.input_hr_stats['mean'], std=self.input_hr_stats['std'])
            print(f"After min-max scaling, using patch-wise normalization for {len(self.selected_bands_hr)} channels for image hr")
            print('=' * 100)
            # norm = PatchwiseNormalize()
            print('Using pre-calculated global mean std')
            transform_list_input_hr.append(norm_hr)
            
        if self.scale_input_hr_to_neg1_1:
            transform_list_input_hr.append(A.Lambda(image=lambda x, **kwargs: x * 2.0 - 1.0))
        transform_list_input_hr.extend([ConvertDtype(),  # Convert float64 to float32 first
                                    ToTensorV2()])
        transform_list_input_hr_compose = A.Compose(transform_list_input_hr, is_check_shapes=False)


        # for output
        scale_target = A.Lambda(image=lambda x, **kwargs: x * 1e+6)
        unit_rescale_target = A.Lambda(image=lambda x, **kwargs: x * self.unit_scale_ratio)

        transform_list_target = []
        if self.use_target_norm:
            norm_target = A.Normalize(mean=self.target_stats['mean'], std=self.target_stats['std'])
            transform_list_target.extend([norm_target, scale_target])
        elif self.use_target_minmax:
            min_max_target = MinMaxScale(self.target_stats["p1"], self.target_stats["p99"])
            transform_list_target.append(min_max_target)
        else:
            transform_list_target.append(unit_rescale_target)
        if self.scale_target_to_neg1_1:
            transform_list_target.append(A.Lambda(image=lambda x, **kwargs: x * 2.0 - 1.0))
        transform_list_target.extend([ConvertDtype(), ToTensorV2()])
        transform_list_target_compose = A.Compose(transform_list_target, is_check_shapes=False)

        return transform_list_input_compose, transform_list_input_hr_compose, transform_list_target_compose

    def __len__(self):
        # Return number of batches we can create
        total_patches = len(self.patch_coords)
        return total_patches // self.batch_size

    def __getitem__(self, idx):

        # Randomly choose a patch size for this batch
        patch_size = random.choice(self.patch_sizes)
        patches = self.patches_by_size[patch_size]
        
        # Get batch_size patches of this size
        start_idx = (idx * self.batch_size) % len(patches)
        end_idx = min(start_idx + self.batch_size, len(patches))
        batch_patches = patches[start_idx:end_idx]
        
        # If we don't have enough patches, wrap around
        if len(batch_patches) < self.batch_size:
            remaining = self.batch_size - len(batch_patches)
            batch_patches.extend(patches[:remaining])
        
        # Load all patches in the batch
        inputs_lr = []
        targets = []
        inputs_hr = []
        
        for patch_data in batch_patches:

            # Last-resort safety net on top of _read_window_with_retry's
            # internal retries: if a patch's read still fails after those
            # retries are exhausted (sustained, not transient, read
            # pressure on its source file), don't let it kill this whole
            # 64-GPU job — swap in a different random patch of the same
            # size and keep going. Bounded so a systemic problem (e.g. a
            # genuinely missing/corrupt file) still surfaces as an error
            # instead of looping forever.
            max_substitutions = 5
            for substitution in range(max_substitutions + 1):
                try:
                    input_patch_lr = LazyPatchDataset._load_multiband_patch(
                        patch_data['input_path'],
                        patch_data['input_crop_y'],
                        patch_data['input_crop_x'],
                        patch_data['input_patch_height'],
                        patch_data['input_patch_width'],
                        self.selected_bands
                    )
                    output_patch = LazyPatchDataset._load_output_patch(
                        patch_data['output_path'],
                        patch_data['output_crop_y'],
                        patch_data['output_crop_x'],
                        patch_data['output_patch_height'],
                        patch_data['output_patch_width']
                    )
                    input_patch_hr = LazyPatchDataset._load_multiband_patch(
                        patch_data['input_path_hr'],
                        patch_data['input_crop_y_hr'],
                        patch_data['input_crop_x_hr'],
                        patch_data['input_patch_height_hr'],
                        patch_data['input_patch_width_hr'],
                        self.selected_bands_hr,
                        ignore_percentile=True
                    )
                    break
                except rasterio.errors.RasterioIOError as e:
                    if substitution == max_substitutions:
                        _log_err(
                            f"{max_substitutions} substitute patches in a "
                            f"row also failed to read; giving up on batch idx={idx}: {e}"
                        )
                        raise
                    _log_err(
                        f"Patch unreadable after retries "
                        f"(input_path={patch_data['input_path']}, "
                        f"input_path_hr={patch_data['input_path_hr']}, "
                        f"output_path={patch_data['output_path']}); "
                        f"substituting a different patch ({substitution + 1}/{max_substitutions}): {e}"
                    )
                    patch_data = random.choice(patches)

            input_transformed = self.transform_input(image=input_patch_lr.astype(np.float32))
            input_patch_lr = input_transformed['image']

            input_transformed_hr = self.transform_input_hr(image=input_patch_hr.astype(np.float32))
            input_patch_hr = input_transformed_hr['image']

            output_transformed = self.transform_output(image=output_patch)
            output_patch = output_transformed['image']
                
            inputs_lr.append(input_patch_lr)
            targets.append(output_patch)
            inputs_hr.append(input_patch_hr)
        
        # Stack into batches
        input_batch = torch.stack(inputs_lr, dim=0)  # (batch_size, C, H, W)
        target_batch = torch.stack(targets, dim=0)  # (batch_size, 1, H, W)
        input_batch_hr = torch.stack(inputs_hr, dim=0).squeeze()

        
        # Set all invalid values to 0 using torch.nan_to_num
        target_batch = torch.nan_to_num(target_batch, nan=0.0, posinf=0.0, neginf=0.0)
        input_batch = torch.nan_to_num(input_batch, nan=0.0, posinf=0.0, neginf=0.0)
        input_batch_hr = torch.nan_to_num(input_batch_hr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scale_input_to_neg1_1:
            input_batch = torch.clamp(input_batch, min=-1.0, max=1.0)
        if self.scale_input_hr_to_neg1_1:
            input_batch_hr = torch.clamp(input_batch_hr, min=-1.0, max=1.0)
        if self.scale_target_to_neg1_1:
            target_batch = torch.clamp(target_batch, min=-1.0, max=1.0)
        else:
            target_batch = torch.clamp(target_batch, min=0.0, max=50)  # for CHM
        return input_batch, input_batch_hr, target_batch