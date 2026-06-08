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

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import pickle
import shutil
import torch
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from marigold import MarigoldDepthPipeline
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.ps_lazydataset import LazyPatchDataset


def _split_coords(all_coords, train_split, val_split, seed):
    n = len(all_coords)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    return (
        [all_coords[i] for i in train_idx],
        [all_coords[i] for i in val_idx],
        [all_coords[i] for i in test_idx],
    )


def _make_dataset(coords, mode, batch_size, cfg_data):
    """Create a LazyPatchDataset from a pre-split coord list."""
    patch_sizes = [(s, s) for s in cfg_data.patch_sizes]
    config = {
        'input_dir': cfg_data.input_dir,
        'input_dir_hr': cfg_data.input_dir_hr,
        'output_dir': cfg_data.output_dir,
        'selected_bands': cfg_data.selected_bands,
        'selected_bands_hr': cfg_data.selected_bands_hr,
        'file_type_input': cfg_data.file_type_input,
        'file_type_output': cfg_data.file_type_output,
        'patch_sizes': patch_sizes,
        'patch_coord_path': cfg_data.patch_coord_path,
        'num_patches_per_tile': cfg_data.num_patches_per_tile,
        'tile_emphasis': cfg_data.tile_emphasis,
        'correlation_cleaning': cfg_data.correlation_cleaning,
        'target_range_edges': cfg_data.target_range_edges,
        'num_patches_per_target_range': cfg_data.num_patches_per_target_range,
        'selected_percentile': cfg_data.selected_percentile,
        'year': cfg_data.year,
        'batch_size': batch_size,
        'mode': mode,
        'use_input_minmax': cfg_data.use_input_minmax,
        'use_input_norm': cfg_data.use_input_norm,
        'input_stats_file': cfg_data.input_stats_file,
        'use_input_hr_minmax': cfg_data.use_input_hr_minmax,
        'use_input_hr_norm': cfg_data.use_input_hr_norm,
        'input_hr_stats_file': cfg_data.input_hr_stats_file,
        'use_target_minmax': cfg_data.use_target_minmax,
        'use_target_norm': cfg_data.use_target_norm,
        'target_stats_file': cfg_data.target_stats_file,
        'unit_scale_ratio': cfg_data.unit_scale_ratio,
        'scale_input_to_neg1_1': cfg_data.get('scale_input_to_neg1_1', False),
        'scale_input_hr_to_neg1_1': cfg_data.get('scale_input_hr_to_neg1_1', False),
        'scale_target_to_neg1_1': cfg_data.get('scale_target_to_neg1_1', False),
    }
    dataset = LazyPatchDataset(config)
    # Override with the pre-split coord subset
    dataset.patch_coords = coords
    dataset.patches_by_size = {}
    for patch_size in patch_sizes:
        dataset.patches_by_size[patch_size] = [
            p for p in coords
            if p['output_patch_height'] == patch_size[0] and p['output_patch_width'] == patch_size[1]
        ]
    print(f"[{mode}] Patches per size after split:")
    for size, patches in dataset.patches_by_size.items():
        print(f"  {size}: {len(patches)} patches")
    return dataset


if "__main__" == __name__:
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Multimodal Training (LR + HR inputs)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/ps_mrg_mm_v0.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config.",
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
        "--base_data_dir", type=str, default="/mnt/data/dataset/marigold/vkitti",
        help="Base path to the datasets.",
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
    if resume_run is not None:
        logging.info(f"Resuming run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        cfg = recursive_load_config(args.config)
        pure_job_name = os.path.basename(args.config).split(".")[0]
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

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

    # -------------------- Logging --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"config: {cfg}")

    if not args.no_wandb:
        if resume_run is not None:
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dict = {"id": wandb_id, "resume": "must", **cfg.wandb}
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

    tb_logger.set_dir(out_dir_tb)
    log_slurm_job_id(step=0)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"device = {device}")

    # -------------------- Snapshot --------------------
    if resume_run is None:
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"Config saved to {_output_path}")
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"Code snapshot saved to: {_code_snapshot_path}")

    # -------------------- Gradient accumulation --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)
    logging.info(
        f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- Data (Multimodal) --------------------
    cfg_data = cfg.dataset

    # Load all patch coordinates from the pkl file
    print(f"Loading patch coordinates from: {cfg_data.patch_coord_path}")
    with open(cfg_data.patch_coord_path, "rb") as f:
        all_coords = pickle.load(f)
    print(f"Total patches loaded: {len(all_coords)}")

    # Split coords into train / val / test
    train_coords, val_coords, test_coords = _split_coords(
        all_coords,
        train_split=cfg_data.train_split,
        val_split=cfg_data.val_split,
        seed=cfg_data.split_seed,
    )
    assert len(set(id(c) for c in train_coords) & set(id(c) for c in val_coords)) == 0

    print(f"Split sizes — train: {len(train_coords)}, val: {len(val_coords)}, test: {len(test_coords)}")
    print("Warning: target statistics (if used) are computed on the full coord set, not excluding test!")

    train_dataset = _make_dataset(train_coords, mode='train', batch_size=eff_bs, cfg_data=cfg_data)
    val_dataset   = _make_dataset(val_coords,   mode='val',   batch_size=1,      cfg_data=cfg_data)
    test_dataset  = _make_dataset(test_coords,  mode='val',   batch_size=1,      cfg_data=cfg_data)

    print(f"Train batches: {len(train_dataset)}")
    print(f"Val batches:   {len(val_dataset)}")
    print(f"Test batches:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # LazyPatchDataset already returns full batches
        shuffle=True,
        num_workers=cfg_data.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg_data.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg_data.workers,
        pin_memory=True,
    )

    # -------------------- Model --------------------
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), **_pipeline_kwargs
    )

    # -------------------- Trainer --------------------
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

    # -------------------- Training --------------------
    try:
        trainer.train(t_end=t_end)
    except Exception as e:
        logging.exception(e)
