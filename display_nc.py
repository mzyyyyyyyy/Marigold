"""
NetCDF 文件结构检查脚本
======================
查看 .nc 文件的维度、坐标、变量、元数据等内部结构，
用于确认下载的 CMIP6 数据格式是否符合预期。

依赖安装：
  pip install xarray netCDF4
"""

import sys
import xarray as xr
import numpy as np

# ============================================================
# 配置：改成你要检查的 nc 文件路径
# ============================================================

NC_PATH = "/mnt/data/china_project/cmip6_data/hurs/hurs_1986_historical_clipped.nc"


def inspect_nc(path):
    print(f"{'='*70}")
    print(f"检查文件：{path}")
    print(f"{'='*70}\n")

    ds = xr.open_dataset(path)

    # ---- 整体概览 ----
    print("【整体结构】")
    print(ds)
    print()

    # ---- 维度详情 ----
    print(f"\n{'='*70}")
    print("【维度（Dimensions）】")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")

    # ---- 坐标详情 ----
    print(f"\n{'='*70}")
    print("【坐标（Coordinates）】")
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        print(f"\n  ▶ {coord_name}")
        print(f"    shape: {coord.shape}")
        print(f"    dtype: {coord.dtype}")
        try:
            vals = coord.values
            if vals.size > 0:
                print(f"    范围: {vals.min()} ~ {vals.max()}")
                print(f"    前3个值: {vals[:3]}")
        except Exception as e:
            print(f"    （无法读取数值: {e}）")
        # 属性（单位、说明等）
        if coord.attrs:
            print(f"    属性: {dict(coord.attrs)}")

    # ---- 数据变量详情 ----
    print(f"\n{'='*70}")
    print("【数据变量（Data Variables）】")
    for var_name in ds.data_vars:
        var = ds.data_vars[var_name]
        print(f"\n  ▶ {var_name}")
        print(f"    维度: {var.dims}")
        print(f"    shape: {var.shape}")
        print(f"    dtype: {var.dtype}")
        if var.attrs:
            print(f"    属性: {dict(var.attrs)}")

        # 抽样统计（避免大文件全量计算太慢，只取第一个时间切片）
        try:
            if "time" in var.dims:
                sample = var.isel(time=0).values
            else:
                sample = var.values
            valid = sample[~np.isnan(sample)] if np.issubdtype(sample.dtype, np.floating) else sample
            if valid.size > 0:
                print(f"    数值范围（首个时间切片）: "
                      f"{np.nanmin(sample):.3f} ~ {np.nanmax(sample):.3f}")
                print(f"    均值: {np.nanmean(sample):.3f}")
        except Exception as e:
            print(f"    （统计失败: {e}）")

    # ---- 全局属性 ----
    print(f"\n{'='*70}")
    print("【全局属性（Global Attributes）】")
    for k, v in ds.attrs.items():
        print(f"  {k}: {v}")

    # ---- 文件大小 ----
    import os
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n{'='*70}")
    print(f"文件大小: {size_mb:.2f} MB")

    ds.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else NC_PATH
    inspect_nc(path)