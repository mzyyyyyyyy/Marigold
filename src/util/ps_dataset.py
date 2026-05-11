import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import rasterio
from rasterio.windows import Window
import random
from tqdm import tqdm
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import concurrent.futures
import multiprocessing
import json
import pickle
import re
import glob
import ipdb
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from shapely.geometry import box
import math

def fourier_encode(coords, num_bands=6, max_freq=10.0):
    """
    coords: shape (2,) normalized to [0,1]
    Returns Fourier features of shape (2*num_bands*2,)
    """
    freqs = np.logspace(0.0, np.log10(max_freq), num=num_bands)  # 1 to max_freq
    coords = coords[:, None]  # shape (2,1)
    x = coords * freqs[None, :] * 2 * np.pi  # shape (2,num_bands)
    fourier_features = np.concatenate([np.sin(x), np.cos(x)], axis=0)  # shape (4,num_bands)
    return fourier_features.flatten()



class SameSizeBatchDataset(Dataset):
    """
    Dataset wrapper that provides batches of the same patch size for efficient GPU utilization
    """
    def __init__(self, base_dataset, patch_sizes: List[Tuple[int, int]], batch_size: int,
                 lons, lats,
                 transform_input: Optional[A.Compose] = None,
                 transform_output: Optional[A.Compose] = None,
                ):
        self.base_dataset = base_dataset

        self.disp_name = getattr(base_dataset, "disp_name", "ps_eur_dataset") # 为 marigold 添加一个属性便于 val
        # self.min_depth = getattr(base_dataset, "min_depth", 0.0)
        # self.max_depth = getattr(base_dataset, "max_depth", 28.0) # marigold 需要 dataset 的极值
        self.filename_ls_path = getattr(base_dataset, "filename_ls_path", None)



        self.patch_sizes = patch_sizes
        self.batch_size = batch_size
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.lons = lons
        self.lats = lats
        self.lons_scaled = (lons-lons.min())/(lons.max()-lons.min()) # tuple showing the range of lon
        self.lats_scaled = (lats-lats.min())/(lats.max()-lats.min())
        self.patch_coordinates = self.get_patch_coordinates()
        
        # Handle different dataset types
        if hasattr(base_dataset, 'get_patches_by_size'):
            # MultiScalePatchDataset - use existing method
            self.patches_by_size = {}
            for patch_size in patch_sizes:
                self.patches_by_size[patch_size] = base_dataset.get_patches_by_size(patch_size, batch_size * 20)
        else:
            # LazyMultiScaleDataset - filter from patch_coordinates
            self.patches_by_size = {}
            for patch_size in patch_sizes:
                # Filter patches by size from patch_coordinates
                size_patches = [
                    patch for patch in self.patch_coordinates
                    if patch['output_patch_height'] == patch_size[0] and patch['output_patch_width'] == patch_size[1]
                ]
                self.patches_by_size[patch_size] = size_patches
        
        print(f"Generated patches for each size:")
        for size, patches in self.patches_by_size.items():
            print(f"  {size}: {len(patches)} patches")
    
    def get_patch_coordinates(self):
        # Handle both Subset and base dataset
        if hasattr(self.base_dataset, 'indices') and hasattr(self.base_dataset, 'dataset'):
            base = self.base_dataset.dataset
            indices = self.base_dataset.indices
            return [base.patch_coordinates[i] for i in indices]
        else:
            return self.base_dataset.patch_coordinates
    
    def __len__(self):
        # Return number of batches we can create
        total_patches = sum(len(patches) for patches in self.patches_by_size.values())
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
        inputs = []
        targets = []
        if self.base_dataset.dataset.input_multimodal:
            inputs_m2 = []

        if self.base_dataset.dataset.use_geo_location:
            input_loc = []
        
        for patch_data in batch_patches:
            # Load patches using the base dataset methods
            base = self.base_dataset
            if hasattr(base, 'dataset') and hasattr(base, 'indices'):
                # It's a Subset
                base = base.dataset


            if self.base_dataset.dataset.use_geo_location:
                # location loading
                input_patches = base.load_multiband_patch(
                    patch_data['input_path'],
                    patch_data['input_crop_y'],
                    patch_data['input_crop_x'],
                    patch_data['input_patch_height'],
                    patch_data['input_patch_width'],
                    self.base_dataset.dataset.selected_bands,
                    ignore_percentile=True,
                    return_tile_id=self.base_dataset.dataset.use_geo_location
                )
                input_patch, loc_patch = input_patches
            else:
                input_patch = base.load_multiband_patch(
                    patch_data['input_path'],
                    patch_data['input_crop_y'],
                    patch_data['input_crop_x'],
                    patch_data['input_patch_height'],
                    patch_data['input_patch_width'],
                    self.base_dataset.dataset.selected_bands,
                    ignore_percentile=True
                )

            
            
            output_patch = base.load_output_patch(
                patch_data['output_path'],
                patch_data['output_crop_y'],
                patch_data['output_crop_x'],
                patch_data['output_patch_height'],
                patch_data['output_patch_width']
            )

            if self.base_dataset.dataset.input_multimodal:
                input_m2_patch = base.load_multiband_patch(
                    patch_data['input_m2_path'],
                    patch_data['input_m2_crop_y'],
                    patch_data['input_m2_crop_x'],
                    patch_data['input_m2_patch_height'],
                    patch_data['input_m2_patch_width'],
                    self.base_dataset.dataset.selected_bands_m2,
                    ignore_percentile=True
                )

            if self.transform_input:
                input_transformed = self.transform_input(image=input_patch.astype(np.float32))
                input_patch = input_transformed['image']

                if self.base_dataset.dataset.input_multimodal:
                    input_m2_transformed = self.transform_input(image=input_m2_patch.astype(np.float32))
                    input_m2_patch = input_m2_transformed['image']

            if self.base_dataset.dataset.use_geo_location:
                lon, lat = loc_patch
                lat_scale = (lat - self.lats.min()) / (self.lats.max() - self.lats.min())
                lon_scale = (lon - self.lons.min()) / (self.lons.max() - self.lons.min())
                coords = np.stack([lon_scale, lat_scale], axis=-1)
                # print(coords)
                coords = fourier_encode(coords)
                # lat_norm = (lat_scale - self.lats_scaled.mean()) / self.lats_scaled.std()
                # lon_norm = (lon_scale - self.lons_scaled.mean()) / self.lons_scaled.std()
                input_loc.append(coords)
                # input_loc.append(loc_patch)
            
            if self.transform_output:
                output_transformed = self.transform_output(image=output_patch)
                output_patch = output_transformed['image']
                
            inputs.append(input_patch)
            targets.append(output_patch)

            if self.base_dataset.dataset.input_multimodal:
                inputs_m2.append(input_m2_patch)
        
        # Stack into batches
        input_batch = torch.stack(inputs, dim=0)  # (batch_size, C, H, W)
        target_batch = torch.stack(targets, dim=0)  # (batch_size, 1, H, W)
        if self.base_dataset.dataset.use_geo_location:
            loc_batch = torch.tensor(np.array(input_loc), dtype=torch.float32) # (b, 2)


        if self.base_dataset.dataset.input_multimodal:
            input_m2_batch = torch.stack(inputs_m2, dim=0).squeeze()
            # print(f'shape is {input_m2_batch.shape}')
        
        # Set all invalid values to 0 or -1 using torch.nan_to_num
        if self.base_dataset.dataset.minmax_target:
            target_batch = torch.nan_to_num(target_batch, nan=-1.0, posinf=-1.0, neginf=-1.0) # if minmax target
        else:
            target_batch = torch.clamp(target_batch, min=0.0, max=30) # if not minmax target, directly clamp to a reasonable range to avoid extreme values, can be tuned based on the data distribution
        
        if self.base_dataset.dataset.use_global_minmax:
            input_batch = torch.nan_to_num(input_batch, nan=-1.0, posinf=-1.0, neginf=-1.0)
        
        # Resample input to match target resolution
        if input_batch.shape[-2:] != target_batch.shape[-2:]:
            input_batch = torch.nn.functional.interpolate(
                input_batch,
                size=target_batch.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        if self.base_dataset.dataset.input_multimodal:
            input_m2_batch = torch.nan_to_num(input_m2_batch, nan=-1.0, posinf=-1.0, neginf=-1.0)
            return (input_batch, input_m2_batch), target_batch
        else:
            if self.base_dataset.dataset.use_geo_location:
                loc_batch = torch.nan_to_num(loc_batch, nan=-1.0, posinf=-1.0, neginf=-1.0)
                # print(loc_batch)
                return (input_batch, loc_batch), target_batch
            else:
                return input_batch, target_batch

class LazyMultiScaleDataset(Dataset):
    """
    Memory-efficient dataset that generates patch coordinates only and loads images on-demand
    """
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 selected_bands: List[int] = [0, 1, 2],
                 file_type_input: str = "jp2",
                 file_type_output: str = "tif",
                 target_name: str = "canopy_height_",
                 patch_sizes: List[Tuple[int, int]] = [(64, 64), (128, 128), (256, 256)],
                 patches_per_tile: int = 100,
                 min_valid_ratio: float = 0.1,
                 normalize_target: bool = False,
                 year: int = 2024,
                 selected_percentile: list[str] = ["p50"],
                 patch_coord_path: str = "",
                 tile_emphasis: list[str] = ['121,30'],
                 nodata_cleaning: bool = False,
                 correlation_cleaning: bool = False,
                 input_dir_m2: str= "",
                 input_multimodal: bool = False,
                 use_geo_location: bool = True,
                 file_type_m2: str = '',
                 selected_bands_m2: list = [0, 1, 2],
                 target_range_edges: list = [0, 100, 200, 300, 400, 1000],
                 num_patches_per_target_range: int = 200,
                 minmax_target: bool = False,
                 use_global_minmax: bool = False,
                 ):
        
        self.input_dir = input_dir
        self.input_dir_m2 = input_dir_m2
        self.input_multimodal = input_multimodal
        self.use_geo_location = use_geo_location
        self.file_type_m2 = file_type_m2
        self.selected_bands_m2 = selected_bands_m2
        self.output_dir = output_dir
        self.selected_bands = selected_bands
        self.input_channels = len(selected_bands)*len(selected_percentile)
        self.input_m2_channels = len(selected_bands_m2)
        self.file_type_input = file_type_input
        self.file_type_output = file_type_output
        self.target_name = target_name
        self.patch_sizes = patch_sizes
        self.patches_per_tile = patches_per_tile
        self.min_valid_ratio = min_valid_ratio
        self.normalize_target = normalize_target
        self.year = year
        self.selected_percentile = selected_percentile
        self.patch_coord_path = patch_coord_path
        self.tile_emphasis = tile_emphasis
        self.correlation_cleaning = correlation_cleaning
        self.target_range_edges = target_range_edges
        self.num_patches_per_target_range = num_patches_per_target_range
        self.minmax_target = minmax_target
        self.use_global_minmax = use_global_minmax


        
        if not os.path.exists(self.patch_coord_path):
            raise ValueError('Please specify a correct path (file or dir) to save the coordinates for training data preparation')

        if os.path.isdir(self.patch_coord_path):
            self.gather_tile_ids()
            if nodata_cleaning:
                self.filter_empty_tiles()
            else:
                print('No no-data cleaning, proceed with all tiles ----')
            self.patch_coordinates = []
            # self._generate_patch_coordinates()
            self._generate_patch_coordinates_parallel()
            try:
                from datetime import datetime
                save_patch_path = os.path.join(self.patch_coord_path, 'save_' + datetime.now().strftime("%Y%m%d") + '.pkl')
                with open(save_patch_path, "wb") as f:
                    pickle.dump(self.patch_coordinates, f)
            except:
                # raise ValueError('Patch coord saving error')
                import ipdb; ipdb.set_trace()
        
        elif os.path.isfile(self.patch_coord_path):
            
            print('------------------------------------')
            print('Loading preprocessed patch coordinates from file path: ', self.patch_coord_path)
            with open(self.patch_coord_path, "rb") as f:
                self.patch_coordinates = pickle.load(f)
                self.check_coord_path_match_config_path()
                self.check_coord_scale_match_data_scale()
            # import ipdb; ipdb.set_trace()
        else:
            raise ValueError('Coord path should be either dir or file')

        if self.input_multimodal:
            print('Processing multimodal data file paths, spatial scale ratio etc.')
            self.prepare_multimodal_data_coord()



    def check_coord_path_match_config_path(self):
        """If coord file is given, double check whether the contained input and output file path match with those given in the
        new config file, otherwise replace the saved path with the new path.

        Useful when coords are saved, but input or output files are changed (without a need to rerun coordinates sampling)
        """
        print('Double checking whether pkl saved input/output paths match with config file paths')
        random_entry = random.choice(self.patch_coordinates)
        # ipdb.set_trace()
        try:
            Path(random_entry.get('input_path')).resolve().relative_to(Path(self.input_dir).resolve())
            input_match = True
        except:
            input_match = False
        try:
            Path(random_entry.get('output_path')).resolve().relative_to(Path(self.output_dir).resolve())
            output_match = True
        except:
            output_match = False
        if input_match and output_match:
            print("A random entry's paths match the config directories.")
            return




        if not input_match:
            print("A random entry's input paths do not match the config directories.")
            # ipdb.set_trace()
            print(f"random entry's paths is {random_entry.get('input_path')}")
            print(f"config dir path is {self.input_dir}")
            print('*' * 10 + 'Rewriting coord path input dirs to config paths')
            pattern = re.compile(r'(\d+,\d+)')
            for element in tqdm(self.patch_coordinates):
                filename = os.path.basename(element['input_path'])
                coordinate = pattern.search(filename).group(1)
                search_pattern = os.path.join(self.input_dir, f"**/*{coordinate}*.{self.file_type_input}")
                found_file = glob.glob(search_pattern, recursive=True)[0]
                assert os.path.exists(found_file)
                element['input_path'] = found_file
        else:
            print("A random entry's output paths do not match the config directories.")
            # ipdb.set_trace()
            print(f"random entry's paths is {random_entry.get('output_path')}")
            print(f"config dir path is {self.output_dir}")
            print('*' * 10 + 'Rewriting coord path output dirs to config paths')
            # ipdb.set_trace()
            pattern = re.compile(r'(\d+,\d+)')
            for element in tqdm(self.patch_coordinates):
                filename = os.path.basename(element['output_path'])
                coordinate = pattern.search(filename).group(1)
                search_pattern = os.path.join(self.output_dir, f"*{coordinate}*.{self.file_type_output}")
                found_file = glob.glob(search_pattern)[0]
                assert os.path.exists(found_file)
                element['output_path'] = found_file

        # ipdb.set_trace()
        
    def check_coord_scale_match_data_scale(self):
        # check whether input/output spatial scale match, otherwise update scale!
        random_entry = random.choice(self.patch_coordinates)
        with rasterio.open(random_entry.get('input_path')) as src:
            input_height, input_width = src.height, src.width
        with rasterio.open(random_entry.get('output_path')) as src:
            output_height, output_width = src.height, src.width

        scale_factor = input_width / output_width

        # scale_factor = round(scale_factor) if (scale_factor>1) else scale_factor
        # # ipdb.set_trace()
        # if not math.isclose(scale_factor, random_entry.get('scale_factor_x'), rel_tol=1e-3, abs_tol=1e-5):
        # # if scale_factor != random_entry.get('scale_factor'):
        #     print('Updating scale factor of input/output')
        #     print(f"calculated is {scale_factor}, but file contained {random_entry.get('scale_factor_x')}")
        #     # update input coord based on output dimension
        #     # note that this will change the sampling distribution! some parts might not be sampled at all
        #     for element in tqdm(self.patch_coordinates):
        #         element['input_crop_y'] = int(element['output_crop_y'] / scale_factor)
        #         element['input_crop_x'] = int(element['output_crop_x'] / scale_factor)
        #         element['input_patch_height'] = int(element['output_patch_height'] / scale_factor)
        #         element['input_patch_width'] = int(element['output_patch_width'] / scale_factor)
        #         element['scale_factor_x'] = scale_factor

        self.in_out_scale_factor = scale_factor
        print(f'Confirmed: Input/Output dimension ratio is {self.in_out_scale_factor:.3f}')



    def prepare_multimodal_data_coord(self):
        # find corresponding multimodal data for each input file; prepare correct cropping location
        # sample a random entry and check if the 2nd modal input has the correct file path
        random_entry = random.choice(self.patch_coordinates)
        try:
            Path(random_entry.get('input_m2_path')).resolve().relative_to(Path(self.input_dir_m2).resolve())
            path2_match = True
        except:
            path2_match = False
        if path2_match:
            return "------- Checked: 2nd modal input data configs match the defined config."
        else:
            print(f'-------- 2nd modal data path not matching specified data path')
            print('-------- Starting 2nd modal data preparation')

            folder2 = Path(self.input_dir_m2)
            from collections import defaultdict
            lookup = defaultdict(list)
            print('pre-indexing 2nd modal data file name')

            for f in tqdm(folder2.rglob(f"*.{self.file_type_m2}")):
                name = f.name
                # Example: file2 format already has zero-padded IDs like 00110_00027
                match2 = re.search(r'(\d{5})_(\d{5})', name)
                if match2:
                    norm_id = f"{match2.group(1)}_{match2.group(2)}"
                    lookup[norm_id].append(f)

            lookup_most_recent = {} # if multiple matches, most recent are the best ones
            for norm_id, paths in lookup.items():
                most_recent_file = max(paths, key=lambda p: p.stat().st_mtime)
                lookup_most_recent[norm_id] = most_recent_file

            print('updating 2nd modal data file name and cropping indices ')
            from concurrent.futures import ProcessPoolExecutor

            def process_all(patch_coordinates, lookup_most_recent, max_workers=12):
                results = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(process_entry, entry, lookup_most_recent)
                        for entry in patch_coordinates
                    ]
                    for f in tqdm(futures):
                        results.append(f.result())  # collect finished results
                return results

            self.patch_coordinates = process_all(self.patch_coordinates, lookup_most_recent, max_workers=max(1, int(multiprocessing.cpu_count() * 0.6)))

            try:
                save_patch_path = self.patch_coord_path.replace('.pkl', '_multimodal.pkl')
                with open(save_patch_path, "wb") as f:
                    pickle.dump(self.patch_coordinates, f)
            except IOError as e:
                # logging.error(f"Failed to save patch coordinates: {e}")
                print(f"Failed to save patch coordinates: {e}")
                raise



    

    def gather_tile_ids(self):
        """Gather all tile IDs from the input directory,
        gather all tile ids from the output directory,
        take the intersection of the two lists to form the tile_ids"""
        self.tile_ids = []
        
        all_input_files = glob.glob(f'{self.input_dir}/**/*{self.selected_percentile[0]}*.{self.file_type_input}') # for landsat
        
        # all_input_files = glob.glob(f'{self.input_dir}/*.{self.file_type_input}') # for planetscope

        all_output_files = glob.glob(f'{self.output_dir}/*.{self.file_type_output}')
        print(f"Found {len(all_input_files)} input files and {len(all_output_files)} output files")

        tiles1 = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0] for file in all_input_files] # for landsat

        # tiles1 = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0:2] for file in all_input_files] # for planetscope
        # tiles1 = [(int(t[0]), int(t[1])) for t in tiles1]
        # tiles1 = [f"{x},{y}" for x, y in tiles1]

        tiles2 = [os.path.basename(file).split(str(self.year) + '_')[1].split('_')[0:2] for file in all_output_files] # for chm
        tiles2 = [(int(t[0]), int(t[1])) for t in tiles2]
        tiles2 = [f"{x},{y}" for x, y in tiles2]

        self.tile_ids = list(set(tiles1) & set(tiles2))
        print(f"After filtering [existing both input and output], found {len(self.tile_ids)} tiles for training and testing!")

    
    def filter_empty_tiles(self):
        """Filter out tiles with invalid pixels"""
        self.target_mean = 0
        self.target_sq = 0
        self.target_npixels = 0
        for tile_id in tqdm(self.tile_ids, desc="Filtering empty tiles and calculating target mean and std"):
            input_path = glob.glob(f'{self.input_dir}/**/*{tile_id}*{self.selected_percentile[0]}*.{self.file_type_input}')[0]
            output_path = glob.glob(f'{self.output_dir}/*{tile_id}*.tif')[0]
            
            with rasterio.open(input_path) as src:
                input_ = src.read()
            with rasterio.open(output_path) as src:
                output = src.read()
                
            # if contains inf or nan, remove the tile
            if np.sum(np.isinf(input_)) > 0 or np.sum(np.isnan(input_)) > 0 or np.sum(np.isinf(output)) > 0 or np.sum(np.isnan(output)) > 0:
                self.tile_ids.remove(tile_id)
            else:
                if self.normalize_target:
                    self.target_npixels, self.target_mean, self.target_sq = update_running_stats(self.target_npixels, self.target_mean, self.target_sq, output)
                
        if self.normalize_target:
            self.target_std = np.sqrt(self.target_sq / (self.target_npixels - 1))
            print(f"Target mean: {self.target_mean}, Target std: {self.target_std}")

        print(f"After filtering invalid tiles, found {len(self.tile_ids)} tiles for training and testing!")



    def _generate_patch_coordinates_parallel(self):
        manager = multiprocessing.Manager()
        progress_counter = manager.Value('i', 0)

        total_patches = len(self.tile_ids) * sum(self.patches_per_tile for _ in self.patch_sizes)  # rough estimate

        all_patch_coords = []
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.7))

        with tqdm(total=total_patches, desc="Processing patches", dynamic_ncols=True) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_tile,
                        tile_id,
                        self.input_dir,
                        self.output_dir,
                        self.selected_percentile,
                        self.file_type_input,
                        self.patch_sizes,
                        self.patches_per_tile,
                        self.load_output_patch,
                        self.load_multiband_patch,
                        progress_counter,
                        self.tile_emphasis,
                        self.selected_bands,
                        self.correlation_cleaning,
                        self.input_multimodal,
                        self.input_dir_m2,
                        self.file_type_m2,
                        self.target_range_edges,
                        self.num_patches_per_target_range
                    )
                    for tile_id in self.tile_ids
                ]

                while any(future.running() for future in futures):
                    # Update the bar based on shared counter
                    pbar.n = progress_counter.value
                    pbar.refresh()

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        all_patch_coords.extend(result)


        self.patch_coordinates = all_patch_coords
        # extract scale factor
        # ipdb.set_trace()
        self.in_out_scale_factor = self.patch_coordinates[0]['scale_factor_x']
        try:
            print('very rough estimate: data left after filtering: ', len(self.patch_coordinates)/total_patches)
        except:
            ipdb.set_trace()




    
    def __len__(self):
        return len(self.patch_coordinates)
    
    def __getitem__(self, idx):
        # defined in another class
        return None, None
    
    def load_multiband_patch(self, filepath: str, y: int, x: int, height: int, width: int, select_bands: list,
                             ignore_percentile = True, return_tile_id = False) -> np.ndarray:
        """Load a patch from a multi-band image with selected bands
            return_tile_id: used to build geo encoders

        """

        if ignore_percentile:
            with rasterio.open(filepath) as src:
                window = Window(x, y, width, height)
                image = src.read(window=window)

                bounds = src.window_bounds(window)

                lon_lat = box(*bounds).centroid.xy
                # print(lon_lat)
                lon, lat = lon_lat[0][0], lon_lat[1][0]

                # Select only the specified bands
                max_band_idx = max(select_bands)
                if max_band_idx >= image.shape[0]:
                    available_bands = list(range(min(len(select_bands), image.shape[0])))
                    selected_image = image[available_bands]
                else:
                    selected_image = image[select_bands]

                images = np.transpose(selected_image, (1, 2, 0))

        else:
            images = []
            for i in range(len(self.selected_percentile)):
                filepath = filepath.replace(self.selected_percentile[0], self.selected_percentile[i])
                with rasterio.open(filepath) as src:
                    window = Window(x, y, width, height)
                    image = src.read(window=window)
                    bounds = src.window_bounds(window)

                    lon_lat = box(*bounds).centroid.xy
                    lon, lat = lon_lat[0][0], lon_lat[1][0]
                    # Select only the specified bands
                    max_band_idx = max(select_bands)
                    if max_band_idx >= image.shape[0]:
                        available_bands = list(range(min(len(select_bands), image.shape[0])))
                        selected_image = image[available_bands]
                    else:
                        selected_image = image[select_bands]

                    image = np.transpose(selected_image, (1, 2, 0))
                    images.append(image)

            images = np.concatenate(images, axis=2)
        if not return_tile_id:
            return images
        else:
            # pattern = re.compile(r'(\d+,\d+)')
            # coordinate = pattern.search(os.path.basename(filepath)).group(1)
            # lon, lat = map(float, coordinate.split(","))
            return (images, [lon, lat])
    
    def load_output_patch(self, filepath: str, y: int, x: int, height: int, width: int) -> np.ndarray:
        """Load a patch from output image"""
        if filepath.endswith('.tif'):
            with rasterio.open(filepath) as src:
                window = Window(x, y, width, height)
                image = src.read(window=window)
                if len(image.shape) == 3:
                    image = image[0]

                # Check for invalid values before any cleaning
                inf_count = np.sum(np.isinf(image))
                nan_count = np.sum(np.isnan(image))
                if inf_count > 0 or nan_count > 0:
                    print(f"Raw data has {inf_count} inf and {nan_count} NaN values")

                # Let the transform handle cleaning instead
                # image[image == np.inf] = 0
                # image[image == -np.inf] = 0
                # image[np.isnan(image)] = 0
                return image
        else:
            image = np.array(Image.open(filepath))
            image = image[y:y+height, x:x+width]
            if len(image.shape) == 3:
                image = image[:, :, 0]
            
            # Check for invalid values before any cleaning
            inf_count = np.sum(np.isinf(image))
            nan_count = np.sum(np.isnan(image))
            if inf_count > 0 or nan_count > 0:
                print(f"Raw data has {inf_count} inf and {nan_count} NaN values")
            
            # Let the transform handle cleaning instead
            # image[image == np.inf] = 0
            # image[image == -np.inf] = 0
            # image[np.isnan(image)] = 0
            return image 
        


def update_running_stats(n, mean, sq, new_val):
    # import ipdb; ipdb.set_trace()
    # clamp new_val to [0, 1000]
    new_val = np.clip(new_val, 0, 1000)
    batch_n = new_val.size
    batch_mean = np.mean(new_val)
    batch_sq =((new_val - batch_mean)**2).sum()
    delta = batch_mean - mean
    total_n = n + batch_n
    new_mean = mean + delta * batch_n / total_n
    new_sq = sq + batch_sq + delta**2 * n * batch_n / total_n  # combined squared deviations
    return total_n, new_mean, new_sq




def process_tile(tile_id, input_dir, output_dir, selected_percentile, file_type, patch_sizes, patches_per_tile,
                 load_output_patch, load_multiband_patch, progress_counter, tile_emphasis, select_bands,
                 correlation_cleaning, input_multimodal, input_dir_m2, file_type_m2,
                 target_range_edges, num_patches_per_target_range):
    """
    Modified to increment progress counter for every patch processed.
    """
    patch_coords = []
    if len(selected_percentile) > 1: # more than one perc
        indc = [14, 17]
    elif len(selected_percentile) == 1:
        indc = [0, 3]

    # tile_id_input = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}" # for planetscope
    # tile_id_output = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}"

    tile_id_output = f"{int(tile_id.split(',')[0]):05d}_{int(tile_id.split(',')[1]):05d}" # for ls

    try:

        input_path = glob.glob(f'{input_dir}/**/*{tile_id}*{selected_percentile[0]}*.{file_type}', recursive=True)[0] # for landsat
        output_path = glob.glob(f'{output_dir}/*{tile_id_output}*.tif')[0] # for chm dataset


        # input_path = glob.glob(f'{input_dir}/*{tile_id_input}*.{file_type}', recursive=True)[0] # for planetscope
        # output_path = glob.glob(f'{output_dir}/*{tile_id_output}*.tif')[0]

        with rasterio.open(input_path) as src:
            input_height, input_width = src.height, src.width
        with rasterio.open(output_path) as src:
            output_height, output_width = src.height, src.width

        scale_factor_x = input_width / output_width # for landsat
        scale_factor_y = input_height / output_height

        # scale_factor_x = 1.0 # for planetscope in EU
        # scale_factor_y = 1.0
        ### TODO: should use different scale factor for x and y, because height/width dimension can be different!
        # ipdb.set_trace()


        if input_multimodal:
            match = re.search(r'_(\d+),(\d+)_', os.path.basename(input_path))
            if not match:
                print(f"Skipping entry for file: {os.path.basename(input_path)} - no ID found for 2nd modal.")
            id_part1 = match.group(1).zfill(5)
            id_part2 = match.group(2).zfill(5)
            search_pattern = f"*{id_part1}_{id_part2}*.{file_type_m2}"
            input_m2_path = glob.glob(os.path.join(input_dir_m2, '**', search_pattern), recursive=True)[0]
            # get rescaling factor from tiles
            # ipdb.set_trace()
            with rasterio.open(input_m2_path) as input2:
                input2_data = input2.read()
                x_m2_factor = input2_data.shape[-1] / input_width
                y_m2_factor = input2_data.shape[-2] / input_height


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
                # print(no_attempt)
                # if no_attempt % 100 == 0:
                #     print(f'After {no_attempt} attempts for this tile, quota is {quota}')

                y_output = random.randint(0, max_y_output)
                x_output = random.randint(0, max_x_output)

                y_input = round(y_output * scale_factor_y)
                x_input = round(x_output * scale_factor_x)

                # y_input = round(y_output * scale_factor_y) - 2 #solve alignment issue by shifting 2 pixels
                # x_input = round(x_output * scale_factor_x) - 2


                input_patch_height = round(patch_height * scale_factor_y)
                input_patch_width = round(patch_width * scale_factor_x)

                if (y_input + input_patch_height > input_height or 
                    x_input + input_patch_width > input_width):
                    continue

                # if (y_input < 0 or x_input < 0): # solve alignment issue by shifting 2 pixels, prevent negative index
                #     continue

                if input_multimodal:
                    # cropping indices for the 2nd modal data
                    y_m2_input = round(y_input * y_m2_factor)
                    x_m2_input = round(x_input * x_m2_factor)
                    input_m2_patch_height = round(input_patch_height * y_m2_factor)
                    input_m2_patch_width = round(input_patch_width * x_m2_factor)

                output_patch = load_output_patch(
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
                    input_patch = load_multiband_patch(
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


                if input_multimodal:
                    patch_coords.append({
                        'tile_id': tile_id,
                        'input_path': input_path,
                        'output_path': output_path,
                        'input_crop_y': y_input,
                        'input_crop_x': x_input,
                        'input_patch_height': input_patch_height,
                        'input_patch_width': input_patch_width,
                        'input_m2_path': input_m2_path,
                        'input_m2_crop_y': y_m2_input,
                        'input_m2_crop_x': x_m2_input,
                        'input_m2_patch_height': input_m2_patch_height,
                        'input_m2_patch_width': input_m2_patch_width,
                        'output_crop_y': y_output,
                        'output_crop_x': x_output,
                        'output_patch_height': patch_height,
                        'output_patch_width': patch_width,
                        'scale_factor_x': scale_factor_x,
                        'scale_factor_y': scale_factor_y
                    })
                else:
                    patch_coords.append({
                        'tile_id': tile_id,
                        'input_path': input_path,
                        'output_path': output_path,
                        'input_crop_y': y_input,
                        'input_crop_x': x_input,
                        'input_patch_height': input_patch_height,
                        'input_patch_width': input_patch_width,
                        'output_crop_y': y_output,
                        'output_crop_x': x_output,
                        'output_patch_height': patch_height,
                        'output_patch_width': patch_width,
                        'scale_factor_x': scale_factor_x,
                        'scale_factor_y': scale_factor_y
                    })
                # ipdb.set_trace()

                progress_counter.value += 1 

    except Exception as e:
        print(f"Error processing tile {tile_id}: {e}")

    return patch_coords


def process_entry(entry, lookup_most_recent):
    filename = Path(entry["input_path"]).name
    match = re.compile(r'_(\d+),(\d+)_').search(filename)
    if not match:
        return entry  # skip if no match

    # Normalize ID
    id_part1 = match.group(1).zfill(5)
    id_part2 = match.group(2).zfill(5)
    norm_id = f"{id_part1}_{id_part2}"

    # Lookup corresponding file
    entry["input_m2_path"] = lookup_most_recent.get(norm_id, None)
    if entry["input_m2_path"] is None:
        return entry  # skip if not found

    # Compare dimensions
    with rasterio.open(entry["input_path"]) as src1, rasterio.open(entry["input_m2_path"]) as src2:
        width1, height1 = src1.width, src1.height
        width2, height2 = src2.width, src2.height

        y_factor = height2 / height1
        x_factor = width2 / width1

        entry['input_m2_crop_y'] = int(entry['input_crop_y'] * y_factor)
        entry['input_m2_crop_x'] = int(entry['input_crop_x'] * x_factor)
        entry['input_m2_patch_height'] = int(entry['input_patch_height'] * y_factor)
        entry['input_m2_patch_width'] = int(entry['input_patch_width'] * x_factor)

    return entry