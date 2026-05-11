
from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from src.util.ps_dataset import *
import torch
import glob
import numpy as np
from tqdm import tqdm
import rasterio
from itertools import product
from rasterio.enums import Resampling
from scipy.ndimage import zoom
import ipdb
from rasterio import windows
from src.util.ps_data_transform import *
from scipy.ndimage import gaussian_filter
from rasterio.transform import from_bounds
import sys
import os
import traceback
from pathlib import Path
import joblib
import json
import re
class Predictor:
    # inference only
    def __init__(self,
                 config,
                 rank = 0
                ):
        self.config = config
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        # for blending
        print('creating blending mask')
        self.blending_mask = create_blending_mask(int(config['data']['patch_size'] * config['data']['out_scale']))
        print('blending mask created with shape: ', self.blending_mask.shape)

        if config['data']['use_gb_minmax'] or config['data']['use_gb_norm']:
            with open(config['data']['globalnorm_stats_file'], "r") as f:
                all_stats = json.load(f)
                try:
                    # ipdb.set_trace()
                    gb_stats = all_stats[str(config['data']['year'])]

                except:
                    print(f'Can not load input statistics for year {config["data"]["year"]}')
        if config['data']['use_gb_minmax']:
            self.min_max = MinMaxScale(gb_stats["p1"], gb_stats["p99"])
        else:
            self.min_max = None
        if config['data']['use_patch_norm']:
            self.patch_norm = PatchwiseNormalize()
        else:
            self.patch_norm = None
        if config['data']['use_gb_norm']:
            self.gb_norm = A.Normalize(mean=gb_stats['mean'], std=gb_stats['std'])
        else:
            self.gb_norm = None

    def build_model(self):

        if self.config['model']['half_precision']:
            dtype = torch.float16
            variant = "fp16"
            logging.warning(
                f"Running with half precision ({dtype}), might lead to suboptimal result."
            )
        else:
            dtype = torch.float32
            variant = None


        checkpoint_path = self.config['model']['model_path']
        self.model : MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
        self.model = self.model.to(self.device)
        logging.info(
            f"Loaded depth pipeline: scale_invariant={self.model.scale_invariant}, shift_invariant={self.model.shift_invariant}"
        )




    def load_checkpoint(self):
        """Load model checkpoint"""
        checkpoint = torch.load(self.config['model']['model_path'], map_location=self.device)

        # # 给模型参数名称不是以 'module.' 开头的加上 'module.' 前缀
        # state_dict = checkpoint['model_state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        
        # for k, v in state_dict.items():
        #     if not k.startswith('module.'):
        #         name = 'module.' + k  # 加上 'module.'
        #     else:
        #         name = k
        #     new_state_dict[name] = v   

        # 不需要添加 'module.' 前缀
        new_state_dict = checkpoint['model_state_dict']

        self.model.load_state_dict(new_state_dict)
        print('*'*20)
        print('Loaded model weights from ', self.config['model']['model_path'])

        if self.config['model']['use_cali_model']:
            self.calibrator = joblib.load(self.config['model']['cali_model_path'])
            print('Loaded calibration weights from ', self.config['model']['cali_model_path'])


    def load_files(self):
        # ipdb.set_trace()
        input_dir = Path(self.config["data"]["input_dir"])


        with open(self.config['data']['grid_id_file'], "r") as f:
            tile_info = json.load(f)
        # ipdb.set_trace()
        geojson_ids = {feature["properties"]["id"] for feature in tile_info["features"]}
        selected_tiles = self.config['data']['selected_test_tile']

        # Decide which IDs to use
        if selected_tiles:  # user explicitly gave IDs in config
            allowed_ids = set(selected_tiles)
        else:  # fallback: use grid file
            allowed_ids = geojson_ids

        filtered_files = []
        for p in input_dir.rglob(f"*{self.config['data']['file_type']}"):

            # filtered_files.append(p) # for ps, no need to filter.

            # for landsat
            name = p.stem  # filename without extension
            if any(tile_id in name for tile_id in allowed_ids): # keep file if it matches one of the allowed IDs
                if (self.config['data']['selected_percentile'][0] in name
                        and self.config['data']['selected_resolution'] in name
                        and not any(excli in name for excli in self.config['data']['exclude_strings'])
                        and not any(excl in p.parts for excl in self.config['data']['exclude_list'])):
                    filtered_files.append(p)


        self.all_files = [(i, i.stem) for i in filtered_files]
        print(f"Selected test tiles: {self.all_files}")
        print('**********************************')
        print('Number of raw image to predict:', len(self.all_files))
        print('Random to-be-predicted file: ', self.all_files[0])

    def predict_all(self):
        is_terminal = sys.stdout.isatty()
        counter = 1
        notwork = []
        outputFiles = []
        year_folder = Path(self.config['data']['input_dir']).name
        model_name = Path(self.config['model']['model_path']).stem
        suffix = self.config['data']['output_suffix'].strip()
        out_dir_full = Path(self.config['data']['output_dir']) / year_folder / f"{model_name}_{suffix}"
        out_dir_full.mkdir(parents=True, exist_ok=True)

        if self.config['model']['use_cali_model']:
            model_set = (self.model, self.calibrator)
            print('Prediting comb: deep model + calibrator ...')
        else:
            model_set = (self.model, None)
            print('Prediction with deep model only ...')

        for fullPath, filename in tqdm(self.all_files, disable=not is_terminal):
            outputFile = out_dir_full / f"{filename}_pred.{self.config['prediction']['file_type']}"

            if not outputFile.exists():

                outputFiles.append(outputFile)

                detectedMask, detectedMeta = predict_img(self.config, model_set, self.device, fullPath, self.blending_mask,
                                                         lons=np.array(self.config['data']['tile_lon_range']),
                                                         lats=np.array(self.config['data']['tile_lat_range']),
                                                         width=self.config['data']['patch_size'], height=self.config['data']['patch_size'],
                                                         stride=self.config['prediction']['stride'], out_scale=self.config['data']['out_scale'],
                                                         min_max = self.min_max, patch_norm = self.patch_norm, gb_norm = self.gb_norm,
                                                         )
                # ipdb.set_trace()
                writeMaskToDisk(detectedMask, detectedMeta, outputFile,
                                image_type=self.config['prediction']['file_type'],
                                data_type = self.config['prediction']['data_type'])

                counter += 1
                # except:
                #     notwork.append(fullPath)

            else:
                print('Skipping: File already analysed!', fullPath)

        print('not working files: ', notwork)
        return out_dir_full



def addTOResult(res, norm_res, bld_mask, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
    bld_mask=bld_mask[:he, :wi]
    valid_mask = (newPredictions > 1).astype(np.float32)
    bld_mask = bld_mask*valid_mask
    # ipdb.set_trace()
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
        res[row:row + he, col:col + wi] = resultant
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
        res[row:row + he, col:col + wi] = resultant
    elif operator == "MIX": # alpha blending # note do not combine with empty regions
        mm1 = currValue!=0
        currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
        mm2 = (currValue==0)
        currValue[mm2] = newPredictions[mm2]
        resultant = currValue
        res[row:row + he, col:col + wi] = resultant
    elif operator == 'blend':
        resultant = newPredictions*bld_mask
        try:
            res[row:row + he, col:col + wi] += resultant
            norm_res[row:row + he, col:col + wi] += bld_mask
        except:
            # ipdb.set_trace()
            print("Error: An error occurred during prediction merge.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    else: #operator == 'REPLACE':
        resultant = newPredictions
        res[row:row + he, col:col + wi] = resultant


    return (res, norm_res)


    
def predict_run(model, device, batch, batch_pos, mask, normmask, bld_mask, batch_loc, operator):

    images = np.stack(batch, axis = 0) #stack a list of arrays along axis
    images = torch.from_numpy(images)
    images = images.to(device, dtype=torch.float)
    deep_model, cali_model = model
    # ipdb.set_trace()
    if batch_loc is not None:
        input_loc = np.stack(batch_loc, axis = 0)
        input_loc = torch.from_numpy(input_loc)
        input_loc = input_loc.to(device, dtype = torch.float)
        input_set = (images, input_loc)
    else:
        input_set = images
    # ipdb.set_trace()
    preds = []
    for i in range(images.shape[0]):
        single = images[i].unsqueeze(0)   # [1, 3, H, W]，值域 [-1, 1]
        single = single[:, :3, :, :] 

        # single = single.half() # 如果half precision, 转为 float16，模型输入要求
        pipe_out = deep_model(
            input_image=single,
            denoising_steps=1,
            processing_res=0  # 不在 pipeline 内部 resize

        )
        pred_single = pipe_out.depth_np  
        pred_single = pred_single * (28.0 - 0.0) + 0.0 # 反归一化
        pred_single = np.clip(pred_single, 0.0, 28.0 * 1.5) # clip to a reasonable range to avoid extreme values, can be tuned based on the data distribution
        preds.append(pred_single)
    pred = np.stack(preds, axis=0)   # [B, H, W]，后续 blending 逻辑不变


    if cali_model:
        pred = cali_model(pred.reshape(-1, 1))
    # ipdb.set_trace()
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        try:
            if pred.ndim == 3:
                p = pred[i]
            else:  # only one patch in this batch
                p = pred

            mask, normmask = addTOResult(mask, normmask, bld_mask, p, row, col, he, wi, operator)
        except:
            print(
                "Error: An error occurred during predict run add to result (possible cause check p = pred[i] no. dimension should be 3 for pred)",
                file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    return mask, normmask


def get_prediction_offsets(
        nrows: int,
        nols: int,
        patch_size: int = 256,
        stride: int = 128
) -> list:
    # Generate offsets for the regular grid
    row_offsets = list(range(0, nrows - patch_size + 1, stride))
    col_offsets = list(range(0, nols - patch_size + 1, stride))

    # Add extra offsets to cover the right and bottom edges
    if (nrows % stride) != 0:
        row_offsets.append(nrows - patch_size)
    if (nols % stride) != 0:
        col_offsets.append(nols - patch_size)

    # Combine all offsets into a single list of (row, col) tuples
    all_offsets = product(col_offsets, row_offsets)

    # ipdb.set_trace()

    return all_offsets


def predict_img(config, model, device, imgPath, bld_mask,
                lons, lats,
                width=256, height=256, stride = 128, out_scale = 0.5,
                min_max = None, patch_norm = None, gb_norm = None,
                ):
    """output has lower resolution, scale = 0.5"""

    with rasterio.open(imgPath) as img:
        nols, nrows = img.meta['width'], img.meta['height']
        meta = img.meta.copy()
        input_bounds = img.bounds
        input_crs = img.crs

        if 'float' not in meta['dtype']:
            meta['dtype'] = np.float32




        # offsets0 = product(range(0, nols, stride), range(0, nrows, stride))
        offsets = get_prediction_offsets(nrows, nols, min(width, height), stride)
        # ipdb.set_trace()
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

        # output mask
        nr_mask = round(nrows*out_scale)
        nc_mask = round(nols*out_scale)
        # ipdb.set_trace()
        masks = np.zeros((nr_mask, nc_mask), dtype=np.float32)
        norm_map = np.zeros_like(masks)


        # # for output
        # output_transform = rasterio.Affine(
        #     input_transform[1] * 2,  # Double the pixel width
        #     input_transform[2],
        #     input_transform[0],
        #     input_transform[3],
        #     input_transform[4],
        #     input_transform[5] * 2  # Double the pixel height
        # )
        output_transform = from_bounds(
            input_bounds.left,
            input_bounds.bottom,
            input_bounds.right,
            input_bounds.top,
            nr_mask,
            nc_mask
        )

        meta.update(
                    {'width': int(nols*out_scale),
                     'height': int(nrows*out_scale),
                     'transform': output_transform,
                     'crs': input_crs
                    }
                    )

        print('double check transformation!')
        print('*'*100)
        # ipdb.set_trace()
        # min-max??
        # transform data
        transform_list = []
        if min_max: # use global min-max scaling
            transform_list.append(min_max)
        if patch_norm:
            transform_list.append(patch_norm)
        if gb_norm:
            transform_list.append(gb_norm)
        transform_list.extend([
            ConvertDtype(),
            # A.Resize(height=height, width=width, interpolation=1),
            ToTensorV2()
        ])

        transform = A.Compose(transform_list, is_check_shapes=False)

        # ipdb.set_trace()
        batch = []
        batch_pos = [ ]

        if config['data']['use_geo_location']:
            # prepare data batch for latlon input
            batch_loc = []
        else:
            batch_loc = None

        # print('prediction start ---')
        for col_off, row_off in tqdm(offsets):

            window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)

            patch = np.zeros((meta['count'], config['data']['patch_size'], config['data']['patch_size']))
            temp_im = img.read(

                out_shape=(
                    img.count,
                    window.height,
                    window.width
                ),
                resampling=Resampling.bilinear, window = window)
            # to filter out sand
            # ipdb.set_trace()
            temp_im = np.transpose(temp_im, axes=(1,2,0))
            # ipdb.set_trace()
            temp_im = transform(image=temp_im.astype(np.float32))['image']


            try:
                patch[..., :window.height, :window.width] = temp_im
                # ipdb.set_trace()
            except:
                print("Error: An error occurred during patch image preparation.", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.exit(1)

            batch.append(patch)
            # for output
            batch_pos.append((int(window.col_off*out_scale), int(window.row_off*out_scale), int(window.width*out_scale), int(window.height*out_scale)))
            # ipdb.set_trace()

            if config['data']['use_geo_location']:
                bounds = img.window_bounds(window)
                # ipdb.set_trace()
                lon_lat = box(*bounds).centroid.xy
                lon, lat = lon_lat[0][0], lon_lat[1][0]
                lat_scale = (lat - lats.min()) / (lats.max() - lats.min())
                lon_scale = (lon - lons.min()) / (lons.max() - lons.min())
                coords = np.stack([lon_scale, lat_scale], axis=-1)
                coords = fourier_encode(coords)
                # loc_x = torch.tensor(np.array([coords]), dtype=torch.float32)
                batch_loc.append(np.array(coords))



            if (len(batch) == config['prediction']['batch_size']):

                curmask = masks[:, :]
                curnorm = norm_map[:, :]

                curmask, curnorm = predict_run(model, device, batch, batch_pos, curmask, curnorm, bld_mask, batch_loc, config['prediction']['operator'])

                batch = []
                batch_pos = []
                if config['data']['use_geo_location']:
                    # prepare data batch for latlon input
                    batch_loc = []
                else:
                    batch_loc = None

        if batch:
            curmask = masks[:, :]
            curnorm = norm_map[:, :]
            curmask, curnorm = predict_run(model, device, batch, batch_pos, curmask, curnorm, bld_mask, batch_loc, config['prediction']['operator'])

            batch = []
            batch_pos = []
            if config['data']['use_geo_location']:
                # prepare data batch for latlon input
                batch_loc = []
            else:
                batch_loc = None
        if config['prediction']['operator'] == 'blend':
            # Normalize the output by dividing by the sum of the weights
            final_output = np.divide(
                masks,
                norm_map,
                out=np.zeros_like(masks),
                where=norm_map != 0
            )
        else:
            final_output = masks

    return final_output, meta




def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, data_type):
    meta = detected_meta.copy()
    # meta['dtype'] =  write_as_type
    meta['count'] = 1
    # ipdb.set_trace()
    if image_type == 'jp2':
        meta.update(
                            {'compress':'lzw',
                              'driver': 'JP2OpenJPEG',
                                'nodata': 65535,
                             'dtype': data_type
                            }
                        )
    elif image_type == 'tif':
        meta.update(
            {'compress': 'lzw',
             'driver': 'GTiff',
             'nodata': 65535,
             'dtype': data_type
             }
        )
    else:
        raise ValueError('check output image type')

    # make Positive
    detected_mask[detected_mask<0] = 0
    # detected_mask = detected_mask.astype(write_as_type)
    assert detected_mask.ndim == 2
    # ipdb.set_trace()
    if data_type == "uint16":
        detected_mask = detected_mask.astype(np.uint16)
    elif data_type == "float32":
        detected_mask = detected_mask.astype(np.float32)
    else:
        raise NotImplementedError('add other data types')
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)

    return




def create_blending_mask(output_patch_size: int) -> np.ndarray:
    """
    Generates a 2D Gaussian blending mask.

    Args:
        output_patch_size: The side length of the model's output patch.

    Returns:
        A NumPy array representing the blending mask.
    """
    mask = np.zeros((output_patch_size, output_patch_size))
    center = output_patch_size / 2
    # A good starting sigma is a quarter of the patch size
    sigma = output_patch_size / 4

    # Place a single point at the center of the mask
    mask[int(center), int(center)] = 1.0

    # Apply a Gaussian filter to create the smooth blending effect
    blending_mask = gaussian_filter(mask, sigma)

    # Normalize the mask so the maximum value is 1.0
    blending_mask /= blending_mask.max()

    return blending_mask