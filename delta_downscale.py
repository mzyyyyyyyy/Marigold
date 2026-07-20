"""
Delta 降尺度脚本：CMIP6 -> CHELSA 分辨率（多情景版本）
========================================================
目标：基于 CMIP6 (0.25°) 与 CHELSA (历史段) 数据，
      生成 2019-2100 年逐年 6 月、CHELSA 分辨率/范围的 7 个气候变量 tif。

Delta 方法：
  1. CMIP6历史基准 = 1986-2018年6月（historical段）均值      (CMIP6原始分辨率)
  2. 对每个未来年份：
       delta = CMIP6未来年6月值 - CMIP6历史基准                (CMIP6原始分辨率)
       delta_highres = 把delta重采样到CHELSA网格                (CHELSA分辨率)
  3. CHELSA基准 = 1986-2018年6月（CHELSA数据）均值              (CHELSA分辨率)
  4. 未来值(CHELSA分辨率) = CHELSA基准 + delta_highres

目录结构（与download_cmip6.py一致）：
  cmip6_data/{var}/{var}_{year}_historical_clipped.nc   (1986-2014，两个情景共用)
  cmip6_data/{var}/{var}_{year}_ssp126_clipped.nc        (2015-2100)
  cmip6_data/{var}/{var}_{year}_ssp585_clipped.nc        (2015-2100)
  chelsa_data/{var}/CHELSA_{var}_{year}_06_clipped.tif   (1986-2018)

输出：
  chelsa_data_future_ssp126/{var}/CHELSA_{var}_{year}_06_clipped.tif
  chelsa_data_future_ssp585/{var}/CHELSA_{var}_{year}_06_clipped.tif

依赖安装：
  pip install xarray netCDF4 rasterio numpy tqdm
"""

import os
import warnings
import numpy as np
import xarray as xr
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 配置参数
# ============================================================

CMIP6_DIR  = "/mnt/data/china_project/cmip6_data"
CHELSA_DIR = "/mnt/data/china_project/chelsa_data"

VARIABLES = ["tas", "tasmin", "tasmax", "pr", "rsds", "hurs", "sfcWind"]

BASELINE_YEAR_START = 1986
BASELINE_YEAR_END   = 2018
FUTURE_YEAR_START   = 2019
FUTURE_YEAR_END     = 2100
TARGET_MONTH        = 6
CMIP6_HIST_END      = 2014   # 2014及以前用historical文件

# 情景 -> 输出目录
SCENARIOS = {
    "ssp126": "/mnt/data/china_project/chelsa_data_future_ssp126",
    "ssp585": "/mnt/data/china_project/chelsa_data_future_ssp585",
}


# ============================================================
# 文件路径函数（与 download_cmip6.py 命名规则完全一致）
# ============================================================

def cmip6_path(var, year, scenario):
    """
    历史段（<=2014）：两个情景共用 historical 文件
    未来段（>2014）：按情景读取对应文件
    """
    if year <= CMIP6_HIST_END:
        fname = f"{var}_{year}_historical_clipped.nc"
    else:
        fname = f"{var}_{year}_{scenario}_clipped.nc"
    return os.path.join(CMIP6_DIR, var, fname)


def chelsa_path(var, year):
    return os.path.join(CHELSA_DIR, var,
                        f"CHELSA_{var}_{year}_{TARGET_MONTH:02d}_clipped.tif")


def output_path(var, year, output_dir):
    return os.path.join(output_dir, var,
                        f"CHELSA_{var}_{year}_{TARGET_MONTH:02d}_clipped.tif")


# ============================================================
# CMIP6 (NetCDF) 处理函数
# ============================================================

def read_cmip6_june_mean(var, year, scenario):
    """读取指定年份CMIP6 nc文件，提取6月逐日数据并求月均值"""
    path = cmip6_path(var, year, scenario)
    ds   = xr.open_dataset(path)

    time_index = ds["time"].values
    months     = np.array([t.month for t in time_index])
    june_mask  = months == TARGET_MONTH

    var_data  = ds[var].values[june_mask]
    june_mean = np.nanmean(var_data, axis=0).astype(np.float32)

    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return june_mean, lon, lat


def build_cmip6_transform(lon, lat):
    """根据CMIP6经纬度坐标构建仿射变换（左上角为原点）"""
    res_lon = lon[1] - lon[0]
    res_lat = lat[1] - lat[0]
    top  = (lat[-1] if lat[0] < lat[-1] else lat[0]) + abs(res_lat) / 2
    left = lon[0] - abs(res_lon) / 2
    return rasterio.transform.from_origin(left, top, abs(res_lon), abs(res_lat))


def cmip6_array_top_down(data, lat):
    """确保数组按纬度从北到南排列（符合栅格惯例）"""
    if lat[0] < lat[-1]:
        data = data[::-1, :]
        lat  = lat[::-1]
    return data, lat


# ============================================================
# 重采样：CMIP6网格 -> CHELSA网格
# ============================================================

def resample_to_chelsa_grid(src_array, src_transform, src_crs,
                            dst_shape, dst_transform, dst_crs):
    dst_array = np.full(dst_shape, np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst_array


# ============================================================
# CHELSA (GeoTIFF) 处理函数
# ============================================================

def read_chelsa_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        meta = {
            "transform": src.transform,
            "crs":       src.crs,
            "shape":     (src.height, src.width),
            "profile":   src.profile.copy(),
        }
    return arr, meta


def save_tif(path, array, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = meta["profile"].copy()
    profile.update({"dtype": "float32", "nodata": np.nan, "compress": "lzw"})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


# ============================================================
# 计算基准（所有情景共用，每个变量只算一次）
# ============================================================

def compute_baselines(var):
    """
    计算 CMIP6 历史基准 和 CHELSA 历史基准。
    historical 段两个情景共用，这里用任意一个情景名（路径逻辑会自动用historical文件）。
    """
    any_scenario = list(SCENARIOS.keys())[0]

    # ---- CMIP6 历史基准 ----
    print(f"  [1/2] CMIP6 历史基准（{BASELINE_YEAR_START}-{BASELINE_YEAR_END}）...")
    cmip6_sum, cmip6_lon, cmip6_lat, n = None, None, None, 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CMIP6历史", leave=False):
        path = cmip6_path(var, year, any_scenario)
        if not os.path.exists(path):
            print(f"    [警告] 缺失：{path}")
            continue

        june_mean, lon, lat   = read_cmip6_june_mean(var, year, any_scenario)
        june_mean, lat_sorted = cmip6_array_top_down(june_mean, lat)

        if cmip6_sum is None:
            cmip6_sum = np.zeros_like(june_mean, dtype=np.float64)
            cmip6_lon, cmip6_lat = lon, lat_sorted

        cmip6_sum += june_mean
        n += 1

    if n == 0:
        print(f"  [错误] {var} 无历史CMIP6数据，跳过")
        return None, None, None, None

    cmip6_baseline  = (cmip6_sum / n).astype(np.float32)
    cmip6_transform = build_cmip6_transform(cmip6_lon, cmip6_lat)
    print(f"    shape={cmip6_baseline.shape}，{n}年")

    # ---- CHELSA 历史基准 ----
    print(f"  [2/2] CHELSA 历史基准（{BASELINE_YEAR_START}-{BASELINE_YEAR_END}）...")
    chelsa_sum, chelsa_meta, n = None, None, 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CHELSA历史", leave=False):
        path = chelsa_path(var, year)
        if not os.path.exists(path):
            print(f"    [警告] 缺失：{path}")
            continue

        arr, meta = read_chelsa_tif(path)
        if chelsa_sum is None:
            chelsa_sum  = np.zeros_like(arr, dtype=np.float64)
            chelsa_meta = meta

        chelsa_sum += np.nan_to_num(arr, nan=0.0)
        n += 1

    if n == 0 or chelsa_meta is None:
        print(f"  [错误] {var} 无历史CHELSA数据，跳过")
        return None, None, None, None

    chelsa_baseline = (chelsa_sum / n).astype(np.float32)
    last_arr, _     = read_chelsa_tif(chelsa_path(var, BASELINE_YEAR_END))
    chelsa_baseline[np.isnan(last_arr)] = np.nan
    print(f"    shape={chelsa_baseline.shape}，{n}年")

    return cmip6_baseline, cmip6_transform, chelsa_baseline, chelsa_meta


# ============================================================
# 核心：单变量单情景的 Delta 降尺度
# ============================================================

def process_scenario(var, scenario, output_dir,
                     cmip6_baseline, cmip6_transform,
                     chelsa_baseline, chelsa_meta):

    dst_shape     = chelsa_meta["shape"]
    dst_transform = chelsa_meta["transform"]
    dst_crs       = chelsa_meta["crs"]
    cmip6_crs     = "EPSG:4326"
    success, fail = 0, []

    for year in tqdm(range(FUTURE_YEAR_START, FUTURE_YEAR_END + 1),
                     desc=f"    [{scenario}] {var}", leave=False):

        out_path = output_path(var, year, output_dir)
        if os.path.exists(out_path):
            success += 1
            continue

        nc_path = cmip6_path(var, year, scenario)
        if not os.path.exists(nc_path):
            fail.append((year, f"文件不存在：{nc_path}"))
            continue

        try:
            future_mean, lon, lat = read_cmip6_june_mean(var, year, scenario)
            future_mean, _        = cmip6_array_top_down(future_mean, lat)

            delta          = future_mean - cmip6_baseline
            delta_highres  = resample_to_chelsa_grid(
                delta, cmip6_transform, "EPSG:4326",
                dst_shape, dst_transform, dst_crs,
            )
            future_highres = chelsa_baseline + delta_highres
            save_tif(out_path, future_highres, chelsa_meta)
            success += 1

        except Exception as e:
            fail.append((year, str(e)))

    print(f"      成功 {success} 个，失败 {len(fail)} 个")
    if fail:
        log_path = os.path.join(output_dir, var, "failed.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            for year, err in fail:
                f.write(f"{year}\t{err}\n")
        for year, err in fail[:5]:
            print(f"      [{year}] {err}")


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"{'#'*60}")
    print(f"# Delta 降尺度：CMIP6 -> CHELSA 分辨率")
    print(f"# 情景：{list(SCENARIOS.keys())}")
    print(f"# 变量：{VARIABLES}")
    print(f"# 基准期：{BASELINE_YEAR_START}-{BASELINE_YEAR_END}")
    print(f"# 预测期：{FUTURE_YEAR_START}-{FUTURE_YEAR_END}")
    print(f"{'#'*60}\n")

    for var in VARIABLES:
        print(f"\n{'='*60}")
        print(f"▶ 变量：{var}")
        print(f"{'='*60}")

        # 基准对所有情景相同，只算一次
        cmip6_baseline, cmip6_transform, chelsa_baseline, chelsa_meta = \
            compute_baselines(var)

        if cmip6_baseline is None:
            continue

        # 对每个情景分别做 Delta 降尺度
        for scenario, output_dir in SCENARIOS.items():
            print(f"\n  ── 情景：{scenario}")
            process_scenario(
                var, scenario, output_dir,
                cmip6_baseline, cmip6_transform,
                chelsa_baseline, chelsa_meta,
            )

    print(f"\n{'#'*60}")
    print("# 全部完成！")
    for scenario, d in SCENARIOS.items():
        print(f"#   {scenario} -> {d}/")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()