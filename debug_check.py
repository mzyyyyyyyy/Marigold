"""
调试脚本2：检查 chelsa_data_future 里的实际输出内容
=====================================================
重点检查：
  1. 不同年份的输出tif是否真的不同
  2. CHELSA历史基准是否计算正确（不应该是0或NaN占大部分）
  3. delta重采样后的数值范围是否合理
"""

import os
import numpy as np
import rasterio

OUTPUT_DIR = "/mnt/data/china_project/chelsa_data_future"   # 改成你的实际路径
CHELSA_DIR = "/mnt/data/china_project/chelsa_data"

VAR = "tas"
TEST_YEARS = [2019, 2030, 2050, 2080, 2100]

print("="*60)
print("【检查A】chelsa_data_future 输出tif的内容对比")
print("="*60)

arrays = {}
for year in TEST_YEARS:
    out_path = os.path.join(OUTPUT_DIR, VAR, f"CHELSA_{VAR}_{year}_06_clipped.tif")
    if not os.path.exists(out_path):
        print(f"  [{year}] 文件不存在：{out_path}")
        continue

    with rasterio.open(out_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr_valid = arr[arr != nodata]
        else:
            arr_valid = arr[~np.isnan(arr)]

        arrays[year] = arr
        print(f"  [{year}] shape={arr.shape}  "
              f"mean={np.nanmean(arr_valid):.6f}  "
              f"min={np.nanmin(arr_valid):.6f}  "
              f"max={np.nanmax(arr_valid):.6f}  "
              f"sample[10,10]={arr[10,10]:.6f}  "
              f"NaN比例={np.isnan(arr).mean()*100:.1f}%")

print("\n  逐年两两对比：")
years_list = list(arrays.keys())
for i in range(len(years_list)):
    for j in range(i+1, len(years_list)):
        y1, y2 = years_list[i], years_list[j]
        identical = np.array_equal(arrays[y1], arrays[y2])
        diff = arrays[y1] - arrays[y2]
        max_diff = np.nanmax(np.abs(diff))
        mean_diff = np.nanmean(np.abs(diff))
        print(f"    {y1} vs {y2}: 完全相同={identical}, "
              f"平均差异={mean_diff:.6f}, 最大差异={max_diff:.6f}")

print("\n" + "="*60)
print("【检查B】对应的CHELSA历史数据（用于对比基准是否合理）")
print("="*60)

for year in [1990, 2000, 2010, 2018]:
    path = os.path.join(CHELSA_DIR, VAR, f"CHELSA_{VAR}_{year}_06_clipped.tif")
    if not os.path.exists(path):
        print(f"  [{year}] 历史文件不存在：{path}")
        continue
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = arr[arr != nodata] if nodata is not None else arr[~np.isnan(arr)]
        print(f"  [{year}] shape={arr.shape}  "
              f"mean={np.nanmean(valid):.6f}  "
              f"min={np.nanmin(valid):.6f}  "
              f"max={np.nanmax(valid):.6f}")

print("\n" + "="*60)
print("【检查C】未来值 vs 历史均值 的差异范围（粗略判断delta是否生效）")
print("="*60)

# 计算CHELSA 1986-2018均值作为粗略基准对比
hist_sum = None
hist_count = 0
for year in range(1986, 2019):
    path = os.path.join(CHELSA_DIR, VAR, f"CHELSA_{VAR}_{year}_06_clipped.tif")
    if not os.path.exists(path):
        continue
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        if hist_sum is None:
            hist_sum = np.zeros_like(arr, dtype=np.float64)
        hist_sum = np.nansum(np.stack([hist_sum, np.nan_to_num(arr, nan=0.0)]), axis=0)
        hist_count += 1

if hist_count > 0:
    hist_mean = hist_sum / hist_count
    print(f"  CHELSA历史均值({hist_count}年): mean={np.nanmean(hist_mean):.6f}, "
          f"sample[10,10]={hist_mean[10,10]:.6f}")

    for year in TEST_YEARS:
        if year in arrays:
            diff_from_hist = arrays[year] - hist_mean
            print(f"  [{year}] 未来值-历史均值: "
                  f"平均={np.nanmean(diff_from_hist):.6f}, "
                  f"sample[10,10]差异={diff_from_hist[10,10]:.6f}")
else:
    print("  [错误] 没有找到历史CHELSA数据")

print("\n调试完成，请把完整输出发给Claude")