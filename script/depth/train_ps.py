# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import os
import shutil
import torch
from torch.utils.data import random_split
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from typing import List, Union
import numpy as np

from marigold import MarigoldDepthPipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.ps_data_transform import get_multiscale_transforms
from src.util.ps_dataset import SameSizeBatchDataset, LazyMultiScaleDataset


if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ps_mrg_v2.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save checkpoints."
    )
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Run without Weights and Biases logging.",
    )
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="On Slurm cluster, do not copy data to the local scratch.",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default="/mnt/data/dataset/marigold/vkitti", help="Base path to the datasets."
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default="/mnt/data/model/marigold/",
        help="Base path to the pretrained checkpoints.",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="Add datetime to the output folder name.",
    )

    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    # -------------------- Initialization --------------------
    # Resume previous run
    if resume_run is not None:
        logging.info(f"Resuming run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Run from start. 从 base_config 递归加载配置，并合并覆盖。一种我之前从来没见过的加载配置的方式。cfg.dataset 就是从这里加载的。
        cfg = recursive_load_config(args.config)
        # Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        # Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        # Output dir
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

    

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- Logging settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    # Initialize wandb
    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dict = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            wandb_cfg_dict = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dict.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dict)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # Tensorboard (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        # Copy and tar code on the first run
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")



    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data_PS --------------------
    cfg_data = cfg.dataset
    patch_sizes=[(size, size) for size in cfg_data.patch_sizes]
    
    if cfg.dataset.efficient_batching:
        print('Using efficient batching')
        base_dataset = LazyMultiScaleDataset(
            input_dir=cfg_data.input_dir, 
            output_dir=cfg_data.output_dir,
            selected_bands=cfg_data.selected_bands,
            file_type_input=cfg_data.file_type_input,
            file_type_output= cfg_data.file_type_output,
            target_name = cfg_data.target_name,
            patch_sizes=patch_sizes,
            patches_per_tile=cfg_data.num_patches_per_tile, 
            min_valid_ratio=cfg_data.min_valid_ratio,
            normalize_target=cfg_data.normalize_target,
            year=cfg_data.year,
            selected_percentile=cfg_data.selected_percentile,
            patch_coord_path = cfg_data.patch_coord_path,
            tile_emphasis = cfg_data.tile_emphasis,
            nodata_cleaning = cfg_data.nodata_cleaning,
            correlation_cleaning = cfg_data.correlation_cleaning,
            input_dir_m2 = cfg_data.input_dir_m2,
            input_multimodal = cfg_data.input_multimodal,
            use_geo_location = cfg_data.use_geo_location,
            file_type_m2 = cfg_data.file_type_m2,
            selected_bands_m2 = cfg_data.selected_bands_m2,
            target_range_edges = cfg_data.target_range_edges,
            num_patches_per_target_range = cfg_data.num_patches_per_target_range
        )

        total_size = len(base_dataset)
        train_size = int(cfg_data.train_split * total_size)
        val_size = int(cfg_data.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            base_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(cfg_data.split_seed)
        )
        
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        test_indices = set(test_dataset.indices)

        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0

        input_channels = base_dataset.input_channels
        in_out_scale_factor = base_dataset.in_out_scale_factor # width, input/output
        # in_out_scale_factor = round(in_out_scale_factor) if (in_out_scale_factor>1) else in_out_scale_factor # for planet
        if cfg_data.input_multimodal:
            input_m2_channels = base_dataset.input_m2_channels

    else:
       raise ValueError('Crop-first batching is not supported yet')

    # collect global norm statistics if using global norm
    if cfg_data.use_global_norm or cfg_data.use_global_minmax:
        with open(cfg_data.globalnorm_stats_file, "r") as f:
            all_stats = json.load(f)
            try:
                # ipdb.set_trace()
                gb_stats = all_stats[str(cfg_data.year)]

            except:
                print(f'Can not load input statistics for year {cfg_data.year}')
    else:
        gb_stats = None

    # collect target norm statistics if using target normalization
    if cfg_data.normalize_target or cfg_data.minmax_target:
        with open(cfg_data.targetnorm_stats_file, "r") as f:
            all_tn_stats = json.load(f)
            try:
                tn_stats = all_tn_stats[str(cfg_data.year)]
            except:
                print(f'Can not load target statistics for year {cfg_data.year}')
    else:
        tn_stats = None


    train_input_transform = get_multiscale_transforms( # 这个函数的作用是根据配置参数创建多尺度数据变换操作的组合。即，train_input_transform是包含多个操作的函数
        patch_sizes, num_channels=input_channels, is_training=True, is_output=False,
        use_global_minmax= cfg_data.use_global_minmax,
        use_global_norm=cfg_data.use_global_norm,
        use_local_norm = cfg_data.use_local_norm,
        globalnorm_stats=gb_stats
    )
    train_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=True, is_output=True, 
        normalize_target=cfg_data.normalize_target,
        minmax_target=cfg_data.minmax_target,
        unit_scale_ratio=cfg_data.unit_scale_ratio,
        target_mean = base_dataset.target_mean if cfg_data.normalize_target else None,
        target_std = base_dataset.target_std if cfg_data.normalize_target else None,
        targetnorm_stats= tn_stats
        # use_shift_augmentation=config['data']['use_shift_augmentation'],
        # shift_limit=config['data']['shift_limit'],
        # shift_augmentation_p=config['data']['shift_augmentation_p']
    )
    val_input_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=False,
        use_global_minmax=cfg_data.use_global_minmax,
        use_global_norm=cfg_data.use_global_norm,
        use_local_norm=cfg_data.use_local_norm,
        globalnorm_stats=gb_stats
    )
    val_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=True,
        normalize_target=cfg_data.normalize_target,
        minmax_target=cfg_data.minmax_target,
        unit_scale_ratio=cfg_data.unit_scale_ratio,
        target_mean = base_dataset.target_mean if cfg_data.normalize_target else None,
        target_std = base_dataset.target_std if cfg_data.normalize_target else None,
        targetnorm_stats= tn_stats
    )
    test_input_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=False,
        use_global_minmax=cfg_data.use_global_minmax,
        use_global_norm=cfg_data.use_global_norm,
        use_local_norm=cfg_data.use_local_norm,
        globalnorm_stats=gb_stats
    )
    test_output_transform = get_multiscale_transforms(
        patch_sizes, num_channels=input_channels, is_training=False, is_output=True,
        normalize_target=cfg_data.normalize_target,
        minmax_target=cfg_data.minmax_target,
        unit_scale_ratio=cfg_data.unit_scale_ratio,
        target_mean = base_dataset.target_mean if cfg_data.normalize_target else None,
        target_std = base_dataset.target_std if cfg_data.normalize_target else None,
        targetnorm_stats= tn_stats
    )

    print('Warning: target mean and std are calculated on the whole dataset (after filtering), not excluding test set!')
    

    # Create efficient batching datasets
    train_dataset = SameSizeBatchDataset(
        train_dataset, 
        patch_sizes, 
        eff_bs,
        transform_input=train_input_transform,
        transform_output=train_output_transform,
        lons=np.array(cfg_data.tile_lon_range),
        lats=np.array(cfg_data.tile_lat_range)
    )
    
    val_dataset = SameSizeBatchDataset(
        val_dataset, 
        patch_sizes, 
        1, # for validation, we can set batch size to 1 in marigold
        transform_input=val_input_transform,
        transform_output=val_output_transform,
        lons=np.array(cfg_data.tile_lon_range),
        lats=np.array(cfg_data.tile_lat_range)
    )
    
    test_dataset = SameSizeBatchDataset(
        test_dataset, 
        patch_sizes, 
        1, # for testing, we can set batch size to 1 in marigold
        transform_input=test_input_transform,
        transform_output=test_output_transform,
        lons=np.array(cfg_data.tile_lon_range),
        lats=np.array(cfg_data.tile_lat_range)
    )
    
    # Create distributed samplers if using multiple GPUs
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # SameSizeBatchDataset already returns batches, iter(dataloader) will reture (1, bs, dimension)
        shuffle=(train_sampler is None), 
        num_workers=cfg_data.workers, 
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # SameSizeBatchDataset already returns batches
        shuffle=(val_sampler is None), # shuffle because not all samples are used in one epoch
        num_workers=cfg_data.workers, 
        pin_memory=True,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # SameSizeBatchDataset already returns batches
        shuffle=(test_sampler is None), # shuffle because not all samples are used in one epoch
        num_workers=cfg_data.workers,
        pin_memory=True,
        sampler=test_sampler
    )


    print(f'Total patches: {len(base_dataset)}')
    print(f'Train patches (len(base_dataset) // batch_size): {len(train_dataset)}')  # len(base_dataset) // batch_size
    print(f'Val patches (len(base_dataset) // batch_size): {len(val_dataset)}')





    # -------------------- Data --------------------


    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), **_pipeline_kwargs
    )

    # -------------------- Trainer --------------------
    # Exit time
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"Will exit at {t_end}")
    else:
        t_end = None

    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"Trainer: {trainer_cls}")
    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        val_dataloaders=[val_loader],
        vis_dataloaders=[test_loader],
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None:
        trainer.load_checkpoint(
            resume_run, load_trainer_state=True, resume_lr_scheduler=True
        )

    # -------------------- Training & Evaluation Loop --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)
