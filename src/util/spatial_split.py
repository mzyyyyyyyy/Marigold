"""Leak-free train/val/test split of LazyPatchDataset patch coordinates.

The old split shuffled the patch list at random, but the pool is made of many
randomly-placed, heavily overlapping crops of a few huge tiles, so overlapping
crops ended up in different splits. Here the split is spatial:

  1. every tile is cut into a regular grid of square blocks (output-pixel
     coordinates),
  2. whole blocks are randomly assigned to train/val/test, per tile, so every
     tile contributes to every split,
  3. a patch belongs to a split only if it lies entirely inside the interior of
     one of that split's blocks (block shrunk by `margin` px on every side).
     Patches crossing a block border are dropped.

train keeps the random (overlapping) crops of the pool that fit; val/test use a
non-overlapping grid of `eval_patch` x `eval_patch` patches laid out inside
their blocks, so every pixel is evaluated at most once. Patches of different
splits are disjoint by construction, with a gap of at least 2 * margin px.
"""

import collections
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SPLITS = ("train", "val", "test")


def patch_rect(p: dict) -> Tuple[int, int, int, int]:
    y0, x0 = p["output_crop_y"], p["output_crop_x"]
    return y0, x0, y0 + p["output_patch_height"], x0 + p["output_patch_width"]


def containing_block(p: dict, block_size: int, margin: int) -> Optional[Tuple[int, int]]:
    """(by, bx) of the block whose margin-shrunk interior fully contains the
    patch, or None if the patch crosses a block border / enters a margin."""
    y0, x0, y1, x1 = patch_rect(p)
    by, bx = y0 // block_size, x0 // block_size
    lo_y, lo_x = by * block_size + margin, bx * block_size + margin
    hi_y, hi_x = (by + 1) * block_size - margin, (bx + 1) * block_size - margin
    if y0 >= lo_y and x0 >= lo_x and y1 <= hi_y and x1 <= hi_x:
        return by, bx
    return None


def assign_blocks(
    blocks_by_tile: Dict[str, List[Tuple[int, int]]],
    train_split: float,
    val_split: float,
    seed: int,
) -> Dict[Tuple[str, int, int], str]:
    """Random per-tile assignment of whole blocks to train/val/test. The test
    fraction is the remainder 1 - train_split - val_split."""
    assert train_split > 0 and val_split > 0 and train_split + val_split < 1
    rng = random.Random(seed)
    block_split = {}
    for tile in sorted(blocks_by_tile):
        blocks = sorted(blocks_by_tile[tile])
        rng.shuffle(blocks)
        n = len(blocks)
        n_train = max(1, round(train_split * n))
        n_val = max(1, round(val_split * n))
        assert n - n_train - n_val >= 1, f"tile {tile}: only {n} blocks, cannot make 3 splits"
        for i, b in enumerate(blocks):
            name = SPLITS[0] if i < n_train else SPLITS[1] if i < n_train + n_val else SPLITS[2]
            block_split[(tile, *b)] = name
    return block_split


def grid_origins_in_block(by: int, bx: int, block_size: int, margin: int, patch: int):
    """Top-left (y, x) of a centred non-overlapping grid of `patch`-sized
    patches inside the block's margin-shrunk interior."""
    interior = block_size - 2 * margin
    n = interior // patch
    assert n >= 1, f"block interior {interior} cannot fit a {patch} patch"
    offset = (interior - n * patch) // 2
    y_start = by * block_size + margin + offset
    x_start = bx * block_size + margin + offset
    return [(y_start + i * patch, x_start + j * patch) for i in range(n) for j in range(n)]


def build_split(
    pool: List[dict],
    train_split: float,
    val_split: float,
    seed: int,
    block_size: int,
    margin: int,
    eval_patch: int,
    make_eval_patch: Callable[[dict, int, int], Optional[dict]],
) -> Tuple[Dict[str, List[dict]], dict]:
    """
    pool: the old patch-coordinate list. Supplies the train patches, the tile
        ids / per-tile templates, and the extent of each tile.
    make_eval_patch(template, y, x) -> coord dict for an `eval_patch` square at
        output pixel (y, x) of the template's tile, or None to reject it
        (out of bounds / mostly empty).
    Returns ({'train','val','test': [coord, ...]}, info).
    """
    assert block_size - 2 * margin >= max(p["output_patch_height"] for p in pool)

    train_pool = collections.defaultdict(list)  # (tile, by, bx) -> [patch]
    n_dropped = 0
    template = {}
    extent = collections.defaultdict(lambda: [0, 0])
    for p in pool:
        template.setdefault(p["tile_id"], p)
        _, _, y1, x1 = patch_rect(p)
        e = extent[p["tile_id"]]
        e[0], e[1] = max(e[0], y1), max(e[1], x1)
        blk = containing_block(p, block_size, margin)
        if blk is None:
            n_dropped += 1
        else:
            train_pool[(p["tile_id"], *blk)].append(p)

    # Blocks covering the tile area reached by the pool (edge blocks may stick
    # out of the tile; their out-of-bounds eval patches are rejected below).
    blocks_by_tile = {
        tile: [(by, bx) for by in range(-(-ey // block_size)) for bx in range(-(-ex // block_size))]
        for tile, (ey, ex) in extent.items()
    }
    block_split = assign_blocks(blocks_by_tile, train_split, val_split, seed)

    out = {s: [] for s in SPLITS}
    n_rejected_eval = 0
    for (tile, by, bx), name in sorted(block_split.items()):
        if name == "train":
            out["train"].extend(train_pool.get((tile, by, bx), []))
        else:
            for y, x in grid_origins_in_block(by, bx, block_size, margin, eval_patch):
                c = make_eval_patch(template[tile], y, x)
                if c is None:
                    n_rejected_eval += 1
                else:
                    out[name].append(c)

    assert_no_overlap_across(out)
    assert_no_overlap_within(out["val"])
    assert_no_overlap_within(out["test"])

    info = {
        "block_size": block_size, "margin": margin, "eval_patch": eval_patch, "seed": seed,
        "train_split": train_split, "val_split": val_split,
        "n_pool_patches": len(pool),
        "n_pool_dropped_crossing_block_border": n_dropped,
        "n_eval_grid_rejected": n_rejected_eval,
        "n_patches": {s: len(out[s]) for s in SPLITS},
        "n_patches_by_tile": {s: dict(collections.Counter(p["tile_id"] for p in out[s])) for s in SPLITS},
        "n_blocks": {s: sum(v == s for v in block_split.values()) for s in SPLITS},
        "blocks": {
            tile: {s: sorted([k[1], k[2]] for k, v in block_split.items() if k[0] == tile and v == s)
                   for s in SPLITS}
            for tile in sorted(blocks_by_tile)
        },
    }
    return out, info


def assert_no_overlap_across(splits: Dict[str, List[dict]], cell: int = 30) -> None:
    """Independent check across splits: rasterise each split's patches on a
    coarse per-tile grid; no cell may be touched by two splits (conservative:
    cells are coarser than a pixel)."""
    owner = {}
    for name, patches in splits.items():
        for p in patches:
            y0, x0, y1, x1 = patch_rect(p)
            for cy in range(y0 // cell, (y1 - 1) // cell + 1):
                for cx in range(x0 // cell, (x1 - 1) // cell + 1):
                    prev = owner.setdefault((p["tile_id"], cy, cx), name)
                    if prev != name:
                        raise AssertionError(
                            f"splits '{prev}' and '{name}' overlap at tile {p['tile_id']} cell ({cy},{cx})"
                        )


def assert_no_overlap_within(patches: List[dict]) -> None:
    """No two patches of one (eval) split may share a pixel."""
    rects = collections.defaultdict(list)
    for p in patches:
        rects[p["tile_id"]].append(patch_rect(p))
    for tile, rs in rects.items():
        arr = np.array(sorted(rs))  # sorted by y0, then x0
        for i in range(len(arr)):
            j = i + 1
            while j < len(arr) and arr[j, 0] < arr[i, 2]:  # y ranges intersect
                if arr[j, 1] < arr[i, 3] and arr[i, 1] < arr[j, 3]:
                    raise AssertionError(f"overlapping eval patches in tile {tile}: {arr[i]} vs {arr[j]}")
                j += 1
