"""
Delta 降尺度脚本：CMIP6 -> CHELSA 年均值版本
==============================================
目标：生成 2019-2100 年逐年年均、CHELSA 分辨率的 7 个气候变量 tif。

Delta 方法：
  1. CMIP6历史基准 = 1986-2018年年均值（从逐日nc提取）  (CMIP6原始分辨率)
  2. 对每个未来年份：
       delta = CMIP6未来年年均值 - CMIP6历史基准          (CMIP6原始分辨率)
       delta_highres = 把delta重采样到CHELSA网格          (CHELSA分辨率)
  3. CHELSA基准 = 1986-2018年均值（已下载好的年均tif）   (CHELSA分辨率)
  4. 未来值 = CHELSA基准 + delta_highres

输入：
  CMIP6_DIR/{var}/{var}_{year}_historical_clipped.nc     (1986-2014)
  CMIP6_DIR/{var}/{var}_{year}_{scenario}_clipped.nc     (2015-2100)
  CHELSA_DIR/{var}/CHELSA_{var}_{year}_annual_clipped.tif (1986-2018)

输出：
  OUTPUT_BASE/chelsa_data_future_{scenario}/{var}/CHELSA_{var}_{year}_annual_clipped.tif

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
CHELSA_DIR = "/mnt/data/china_project/chelsa_data_annual"   # 年均tif所在目录

VARIABLES = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]

BASELINE_YEAR_START = 1986
BASELINE_YEAR_END   = 2018
FUTURE_YEAR_START   = 2019
FUTURE_YEAR_END     = 2100
CMIP6_HIST_END      = 2014   # 2014及以前用historical文件

# 三种情景 -> 各自输出目录
SCENARIOS = {
    "ssp126": "/mnt/data/china_project/chelsa_data_future_annual_ssp126",
    "ssp245": "/mnt/data/china_project/chelsa_data_future_annual_ssp245",
    "ssp585": "/mnt/data/china_project/chelsa_data_future_annual_ssp585",
}


# ============================================================
# 文件路径函数
# ============================================================

def cmip6_path(var, year, scenario):
    if year <= CMIP6_HIST_END:
        fname = f"{var}_{year}_historical_clipped.nc"
    else:
        fname = f"{var}_{year}_{scenario}_clipped.nc"
    return os.path.join(CMIP6_DIR, var, fname)


def chelsa_path(var, year):
    return os.path.join(CHELSA_DIR, var,
                        f"CHELSA_{var}_{year}_annual_clipped.tif")


def output_path(var, year, output_dir):
    return os.path.join(output_dir, var,
                        f"CHELSA_{var}_{year}_annual_clipped.tif")


# ============================================================
# CMIP6 处理函数：提取全年均值
# ============================================================

def read_cmip6_annual_mean(var, year, scenario):
    """
    读取 CMIP6 nc 文件，对全年逐日数据求年均值。
    pr（降水）求年总量，其余变量求年均值（与CHELSA年均处理方式一致）。
    """
    path = cmip6_path(var, year, scenario)
    ds   = xr.open_dataset(path)

    var_data = ds[var].values   # (365, lat, lon)

    if var == "pr":
        # 降水：日总量 × 天数 → 年总量
        annual = np.nansum(var_data, axis=0).astype(np.float32)
    else:
        annual = np.nanmean(var_data, axis=0).astype(np.float32)

    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return annual, lon, lat


def build_cmip6_transform(lon, lat):
    res_lon = lon[1] - lon[0]
    res_lat = lat[1] - lat[0]
    top  = (lat[-1] if lat[0] < lat[-1] else lat[0]) + abs(res_lat) / 2
    left = lon[0] - abs(res_lon) / 2
    return rasterio.transform.from_origin(left, top, abs(res_lon), abs(res_lat))


def cmip6_array_top_down(data, lat):
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
    profile.update({"dtype": "float32", "nodata": np.nan,
                    "compress": "lzw", "count": 1})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


# ============================================================
# 计算基准（所有情景共用，每个变量只算一次）
# ============================================================

def compute_baselines(var):
    any_scenario = list(SCENARIOS.keys())[0]

    # ---- CMIP6 历史年均基准 ----
    print(f"  [1/2] CMIP6 历史基准（{BASELINE_YEAR_START}-{BASELINE_YEAR_END}，年均值）...")
    cmip6_sum, cmip6_lon, cmip6_lat, n = None, None, None, 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CMIP6", leave=False):
        path = cmip6_path(var, year, any_scenario)
        if not os.path.exists(path):
            print(f"    [警告] 缺失：{path}")
            continue

        annual, lon, lat   = read_cmip6_annual_mean(var, year, any_scenario)
        annual, lat_sorted = cmip6_array_top_down(annual, lat)

        if cmip6_sum is None:
            cmip6_sum = np.zeros_like(annual, dtype=np.float64)
            cmip6_lon, cmip6_lat = lon, lat_sorted

        cmip6_sum += annual
        n += 1

    if n == 0:
        print(f"  [错误] {var} 无历史CMIP6数据")
        return None, None, None, None

    # CMIP6历史基准：对多年年均值再取均值（pr也是，因为每年已经是年总量，再取均值得到"典型年总量"）
    cmip6_baseline  = (cmip6_sum / n).astype(np.float32)
    cmip6_transform = build_cmip6_transform(cmip6_lon, cmip6_lat)
    print(f"    shape={cmip6_baseline.shape}，{n}年均值")

    # ---- CHELSA 历史年均基准 ----
    print(f"  [2/2] CHELSA 历史基准（{BASELINE_YEAR_START}-{BASELINE_YEAR_END}，年均tif）...")
    chelsa_sum, chelsa_meta, n = None, None, 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CHELSA", leave=False):
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
        print(f"  [错误] {var} 无历史CHELSA数据")
        return None, None, None, None

    chelsa_baseline = (chelsa_sum / n).astype(np.float32)
    last_arr, _     = read_chelsa_tif(chelsa_path(var, BASELINE_YEAR_END))
    chelsa_baseline[np.isnan(last_arr)] = np.nan
    print(f"    shape={chelsa_baseline.shape}，{n}年均值")

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
            future_annual, lon, lat = read_cmip6_annual_mean(var, year, scenario)
            future_annual, _        = cmip6_array_top_down(future_annual, lat)

            if var == "pr":
                # pr: CMIP6 单位为 kg/m²/s，CHELSA 单位为 mm/year，两者量纲不同。
                # 用乘法 delta（比值法）消除单位依赖，同时避免产生负降水。
                ratio = future_annual / np.where(cmip6_baseline == 0, np.nan, cmip6_baseline)
                signal_highres = resample_to_chelsa_grid(
                    ratio, cmip6_transform, "EPSG:4326",
                    dst_shape, dst_transform, dst_crs,
                )
                future_highres = chelsa_baseline * signal_highres
                future_highres = np.maximum(future_highres, 0)
            else:
                # 其余变量加法 delta，CMIP6 与 CHELSA 单位相同：
                #   tas/tasmax/tasmin : K差 = °C差，兼容
                #   hurs  : % (CMIP6) vs % (CHELSA after ×0.01)
                #   rsds  : W/m²
                #   sfcWind: m/s
                delta         = future_annual - cmip6_baseline
                delta_highres = resample_to_chelsa_grid(
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
    print(f"# Delta 降尺度：CMIP6 -> CHELSA 年均值")
    print(f"# 情景：{list(SCENARIOS.keys())}")
    print(f"# 变量：{VARIABLES}")
    print(f"# 基准期：{BASELINE_YEAR_START}-{BASELINE_YEAR_END}")
    print(f"# 预测期：{FUTURE_YEAR_START}-{FUTURE_YEAR_END}")
    print(f"{'#'*60}\n")

    for var in VARIABLES:
        print(f"\n{'='*60}")
        print(f"▶ 变量：{var}")
        print(f"{'='*60}")

        cmip6_baseline, cmip6_transform, chelsa_baseline, chelsa_meta = \
            compute_baselines(var)

        if cmip6_baseline is None:
            print(f"  跳过 {var}")
            continue

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