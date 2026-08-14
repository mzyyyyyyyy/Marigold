"""
修正 CHELSA 年均 tif 单位
=========================
CHELSA V2.1 原始数据存储时有 scale factor，下载脚本未正确应用：

  tas/tasmax/tasmin : 原始值为 0.1K，下载时错误地直接 -273.15
                      正确应为 raw × 0.1 - 273.15
                      修正公式：(stored + 273.15) × 0.1 - 273.15  → °C

  hurs, pr, rsds, sfcWind : scale factor = 0.01，下载时未应用
                             修正公式：stored × 0.01            → 标准物理单位

运行后：
  - 历史年均 tif 就地修正，单位变为标准物理量
  - 修正后即可用于 Delta 降尺度

依赖：pip install rasterio numpy tqdm
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

CHELSA_DIR = "/mnt/data/china_project/chelsa_data_annual"

VARIABLES = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]

TEMP_VARS = {"tas", "tasmax", "tasmin"}


def correct_array(arr, var):
    """将存储值转换为正确物理单位"""
    if var in TEMP_VARS:
        # stored = raw_0.1K - 273.15  →  correct_°C = (stored + 273.15) × 0.1 - 273.15
        return (arr + 273.15) * 0.1 - 273.15
    else:
        return arr * 0.01


def fix_tif(path, var):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None and not np.isnan(nodata):
        mask = arr == nodata
    else:
        mask = np.isnan(arr)

    arr[~mask] = correct_array(arr[~mask], var)
    arr[mask] = np.nan

    profile.update({"dtype": "float32", "nodata": np.nan, "compress": "lzw"})

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def main():
    print("=" * 60)
    print("修正 CHELSA 年均 tif 单位")
    print("=" * 60)

    for var in VARIABLES:
        var_dir = os.path.join(CHELSA_DIR, var)
        files = sorted(glob.glob(os.path.join(var_dir, "*.tif")))
        if not files:
            print(f"\n{var}: 无文件，跳过")
            continue

        print(f"\n▶ {var} ({len(files)} 个文件)")
        for path in tqdm(files, desc=f"  {var}"):
            fix_tif(path, var)

        # 打印修正后样本值
        sample = files[len(files) // 2]
        with rasterio.open(sample) as src:
            arr = src.read(1).astype(np.float32)
        print(f"  修正后抽样 ({os.path.basename(sample)}): mean={np.nanmean(arr):.3f}")

    print("\n完成！请删除已生成的未来数据后重新运行 Delta_downscale_annual.py")


if __name__ == "__main__":
    main()
