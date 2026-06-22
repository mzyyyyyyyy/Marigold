"""
Delta 降尺度脚本：CMIP6 -> CHELSA 分辨率
==========================================
目标：基于 CMIP6 (0.25°) 与 CHELSA (历史段) 数据，
      生成 2019-2100 年逐年 6 月、CHELSA 分辨率/范围的 7 个气候变量 tif。

Delta 方法：
  1. CMIP6历史基准 = 1986-2018年6月（historical段）均值      (CMIP6原始分辨率)
  2. 对每个未来年份：
       delta = CMIP6未来年6月值 - CMIP6历史基准                (CMIP6原始分辨率)
       delta_highres = 把delta重采样到CHELSA网格                (CHELSA分辨率)
  3. CHELSA基准 = 1986-2018年6月（CHELSA数据）均值              (CHELSA分辨率)
  4. 未来值(CHELSA分辨率) = CHELSA基准 + delta_highres

输入：
  CMIP6_DIR/{var}/{var}_{year}_historical_clipped.nc   (1986-2014)
  CMIP6_DIR/{var}/{var}_{year}_ssp245_clipped.nc        (2015-2100)
  CHELSA_DIR/{var}/CHELSA_{var}_{year}_06_clipped.tif   (1986-2018)

输出：
  OUTPUT_DIR/{var}/CHELSA_{var}_{year}_06_clipped.tif   (2019-2100)
  文件名格式与CHELSA历史数据保持一致，方便后续脚本直接复用

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
CHELSA_DIR = "/mnt/data/china_project/chelsa_data"     # 已有历史数据目录
OUTPUT_DIR = "/mnt/data/china_project/chelsa_data_future"      # 未来数据直接存进同一目录的同名子文件夹

VARIABLES = ["tas", "tasmin", "tasmax", "pr", "rsds", "hurs", "sfcWind"]

# 用于计算Delta基准的历史时段（CHELSA与CMIP6均用此范围求均值）
BASELINE_YEAR_START = 1986
BASELINE_YEAR_END   = 2018

# 未来预测时段
FUTURE_YEAR_START = 2019
FUTURE_YEAR_END   = 2100

TARGET_MONTH = 6   # 只处理6月

# CMIP6 historical/ssp245 分段年份（2014及以前用historical文件，之后用ssp245文件）
CMIP6_HIST_END = 2014


# ============================================================
# 文件路径函数
# ============================================================

def cmip6_path(var, year):
    """返回该年对应的CMIP6 nc文件路径（自动判断historical/ssp245）"""
    scenario = "historical" if year <= CMIP6_HIST_END else "ssp245"
    return os.path.join(CMIP6_DIR, var, f"{var}_{year}_{scenario}_clipped.nc")


def chelsa_path(var, year):
    return os.path.join(CHELSA_DIR, var, f"CHELSA_{var}_{year}_{TARGET_MONTH:02d}_clipped.tif")


def output_path(var, year):
    return os.path.join(OUTPUT_DIR, var, f"CHELSA_{var}_{year}_{TARGET_MONTH:02d}_clipped.tif")


# ============================================================
# CMIP6 (NetCDF) 处理函数
# ============================================================

def read_cmip6_june_mean(var, year):
    """
    读取指定年份的CMIP6 nc文件，提取6月份逐日数据并求月均值。
    返回：
      data (lat, lon) 的 numpy 数组
      lon, lat 坐标数组（用于构建transform）
    """
    path = cmip6_path(var, year)
    ds = xr.open_dataset(path)

    # 时间坐标可能是cftime对象，用month属性筛选6月
    time_index = ds["time"].values
    months = np.array([t.month for t in time_index])
    june_mask = months == TARGET_MONTH

    var_data = ds[var].values[june_mask]   # (n_june_days, lat, lon)
    june_mean = np.nanmean(var_data, axis=0).astype(np.float32)   # (lat, lon)

    lon = ds["lon"].values
    lat = ds["lat"].values

    ds.close()
    return june_mean, lon, lat


def build_cmip6_transform(lon, lat):
    """
    根据CMIP6的经纬度坐标构建仿射变换(transform)。
    CMIP6数据通常是等间隔网格，左上角对应 lat[0](最大纬度,需确认排列)和lon[0]。
    """
    res_lon = lon[1] - lon[0]
    res_lat = lat[1] - lat[0]

    # 判断纬度是升序还是降序排列，统一为"左上角原点"格式
    if lat[0] < lat[-1]:
        # 升序：lat[0]是最小值，需要反转，左上角应为最大纬度
        top = lat[-1] + abs(res_lat) / 2
    else:
        top = lat[0] + abs(res_lat) / 2

    left = lon[0] - abs(res_lon) / 2

    transform = rasterio.transform.from_origin(left, top, abs(res_lon), abs(res_lat))
    return transform


def cmip6_array_top_down(data, lat):
    """
    确保数据数组按"纬度从北到南"（即从上到下，符合栅格惯例）排列。
    若lat是升序（南->北），需要把数据和lat都翻转。
    """
    if lat[0] < lat[-1]:
        data = data[::-1, :]
        lat = lat[::-1]
    return data, lat


# ============================================================
# 重采样函数：把CMIP6网格的数据重采样到CHELSA网格
# ============================================================

def resample_to_chelsa_grid(src_array, src_transform, src_crs,
                            dst_shape, dst_transform, dst_crs,
                            resampling=Resampling.bilinear):
    """
    把任意分辨率的数组重采样到目标(CHELSA)网格。
    """
    dst_array = np.full(dst_shape, np.nan, dtype=np.float32)

    rasterio.warp.reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst_array


# ============================================================
# CHELSA (GeoTIFF) 处理函数
# ============================================================

def read_chelsa_tif(path):
    """读取CHELSA tif，nodata转NaN，返回数组及网格信息"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": (src.height, src.width),
            "profile": src.profile.copy(),
        }
    return arr, meta


def save_tif(path, array, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = meta["profile"].copy()
    profile.update({
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


# ============================================================
# 核心：计算单个变量的 Delta 降尺度
# ============================================================

def process_variable(var):
    print(f"\n{'='*60}")
    print(f"▶ 处理变量：{var}")
    print(f"{'='*60}")

    # ---------------------------------------------------
    # 第一步：CMIP6历史基准（1986-2018年6月均值，CMIP6原始网格）
    # ---------------------------------------------------
    print(f"  [1/4] 计算 CMIP6 历史基准（{BASELINE_YEAR_START}-{BASELINE_YEAR_END}年6月均值）...")

    cmip6_baseline_sum = None
    cmip6_lon, cmip6_lat = None, None
    n_years = 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CMIP6历史基准", leave=False):
        path = cmip6_path(var, year)
        if not os.path.exists(path):
            print(f"    [警告] 缺失文件，跳过：{path}")
            continue

        june_mean, lon, lat = read_cmip6_june_mean(var, year)
        june_mean, lat_sorted = cmip6_array_top_down(june_mean, lat)

        if cmip6_baseline_sum is None:
            cmip6_baseline_sum = np.zeros_like(june_mean, dtype=np.float64)
            cmip6_lon, cmip6_lat = lon, lat_sorted

        cmip6_baseline_sum += june_mean
        n_years += 1

    if n_years == 0:
        print(f"  [错误] {var} 没有任何历史CMIP6数据，跳过该变量")
        return

    cmip6_baseline = (cmip6_baseline_sum / n_years).astype(np.float32)
    cmip6_transform = build_cmip6_transform(cmip6_lon, cmip6_lat)
    cmip6_crs = "EPSG:4326"

    print(f"    CMIP6基准网格 shape: {cmip6_baseline.shape}，"
          f"历史年数: {n_years}")

    # ---------------------------------------------------
    # 第二步：CHELSA历史基准（1986-2018年6月均值，CHELSA网格）
    # ---------------------------------------------------
    print(f"  [2/4] 计算 CHELSA 历史基准（同期均值，CHELSA高分辨率网格）...")

    chelsa_baseline_sum = None
    chelsa_meta = None
    n_chelsa_years = 0

    for year in tqdm(range(BASELINE_YEAR_START, BASELINE_YEAR_END + 1),
                     desc="    CHELSA历史基准", leave=False):
        path = chelsa_path(var, year)
        if not os.path.exists(path):
            print(f"    [警告] 缺失文件，跳过：{path}")
            continue

        arr, meta = read_chelsa_tif(path)

        if chelsa_baseline_sum is None:
            chelsa_baseline_sum = np.zeros_like(arr, dtype=np.float64)
            chelsa_meta = meta

        # 处理NaN：用0累加但记录有效计数，最后再做掩膜处理（简化：假定NaN位置在历年一致）
        valid = ~np.isnan(arr)
        if "valid_count" not in dir():
            pass
        chelsa_baseline_sum = np.nansum(
            np.stack([chelsa_baseline_sum, np.nan_to_num(arr, nan=0.0)]), axis=0
        )
        n_chelsa_years += 1

    if n_chelsa_years == 0 or chelsa_meta is None:
        print(f"  [错误] {var} 没有任何CHELSA历史数据，跳过该变量")
        return

    chelsa_baseline = (chelsa_baseline_sum / n_chelsa_years).astype(np.float32)

    # 重新获取一份带正确NaN掩膜的基准（用最后一年的有效性掩膜近似）
    last_year_arr, _ = read_chelsa_tif(chelsa_path(var, BASELINE_YEAR_END))
    chelsa_baseline[np.isnan(last_year_arr)] = np.nan

    print(f"    CHELSA基准网格 shape: {chelsa_baseline.shape}，"
          f"历史年数: {n_chelsa_years}")

    dst_shape     = chelsa_meta["shape"]
    dst_transform = chelsa_meta["transform"]
    dst_crs       = chelsa_meta["crs"]

    # ---------------------------------------------------
    # 第三步 & 第四步：逐年计算delta，重采样，叠加，保存
    # ---------------------------------------------------
    print(f"  [3/4] 逐年计算未来气候（{FUTURE_YEAR_START}-{FUTURE_YEAR_END}）...")

    success = 0
    fail = []

    for year in tqdm(range(FUTURE_YEAR_START, FUTURE_YEAR_END + 1),
                     desc=f"    {var} 未来年份"):
        out_path = output_path(var, year)
        if os.path.exists(out_path):
            success += 1
            continue

        nc_path = cmip6_path(var, year)
        if not os.path.exists(nc_path):
            fail.append((year, "CMIP6文件不存在"))
            continue

        try:
            # 读取该年CMIP6 6月均值
            future_mean, lon, lat = read_cmip6_june_mean(var, year)
            future_mean, _ = cmip6_array_top_down(future_mean, lat)

            # 计算delta（CMIP6原始分辨率）
            delta = future_mean - cmip6_baseline

            # 重采样delta到CHELSA网格
            delta_highres = resample_to_chelsa_grid(
                delta, cmip6_transform, cmip6_crs,
                dst_shape, dst_transform, dst_crs,
            )

            # 叠加到CHELSA基准
            future_highres = chelsa_baseline + delta_highres

            # 保存
            save_tif(out_path, future_highres, chelsa_meta)
            success += 1

        except Exception as e:
            fail.append((year, str(e)))

    # ---------------------------------------------------
    # 汇总
    # ---------------------------------------------------
    print(f"  [4/4] 完成：成功 {success} 个，失败 {len(fail)} 个")
    if fail:
        for year, err in fail[:10]:
            print(f"    [{year}] 失败：{err}")
        if len(fail) > 10:
            print(f"    ...还有 {len(fail)-10} 个失败，详见日志")

        log_path = os.path.join(OUTPUT_DIR, var, "delta_failed.txt")
        with open(log_path, "w") as f:
            for year, err in fail:
                f.write(f"{year}\t{err}\n")


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"{'#'*60}")
    print(f"# Delta 降尺度：CMIP6 -> CHELSA")
    print(f"# 变量：{VARIABLES}")
    print(f"# 历史基准期：{BASELINE_YEAR_START}-{BASELINE_YEAR_END}")
    print(f"# 未来预测期：{FUTURE_YEAR_START}-{FUTURE_YEAR_END}")
    print(f"# 月份：{TARGET_MONTH}")
    print(f"{'#'*60}")

    for var in VARIABLES:
        process_variable(var)

    print(f"\n{'#'*60}")
    print(f"# 全部完成！输出目录：{OUTPUT_DIR}/")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
