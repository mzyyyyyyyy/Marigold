"""
NASA NEX-GDDP-CMIP6 下载脚本（GFDL-ESM4，SSP2-4.5）
====================================================
下载范围：1986—2100年，云南/湖南/广西/贵州研究区
变量：tas, tasmin, tasmax, pr, rsds, hurs, sfcWind

数据分两段：
  historical : 1986—2014（与CHELSA重叠，用于计算Delta基准）
  ssp245     : 2015—2100（未来推理）

两种下载模式（自动切换）：
  模式一【优先】：NCSS远程裁剪，只下载研究区数据，不下载全球文件
  模式二【备用】：下载全球NetCDF文件，本地用xarray裁剪后删除原始文件

依赖安装：
  pip install requests netCDF4 xarray tqdm
"""

import os
import requests
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 配置参数
# ============================================================

MODEL = "GFDL-ESM4"
ENSEMBLE = "r1i1p1f1"
GRID_LABEL = "gr1"   # 如遇404，改为 "gn" 后重试（不同模式网格标签不同）

VARIABLES = ["tas", "tasmin", "tasmax", "pr", "rsds", "hurs", "sfcWind"]

BBOX = {
    "west":  96.9999998617999921,
    "south": 19.9997332551999989,
    "east":  115.0000831849999940,
    "north": 31.0002333326000006,
}

YEAR_START = 1986
YEAR_END   = 2100
HISTORICAL_END = 2014

OUTPUT_DIR = "./cmip6_data"
TEMP_DIR   = "./cmip6_temp_global"   # 备用模式的全球文件临时存放

THREDDS_BASE = "https://ds.nccs.nasa.gov/thredds"
NCSS_BASE    = f"{THREDDS_BASE}/ncss/grid/AMES/NEX/GDDP-CMIP6"
FILESERVER_BASE = f"{THREDDS_BASE}/fileServer/AMES/NEX/GDDP-CMIP6"

MAX_CONSEC_FAIL = 3   # NCSS连续失败超过此数，切换为下载模式


# ============================================================
# 工具函数
# ============================================================

def scenario_for_year(year):
    return "historical" if year <= HISTORICAL_END else "ssp245"


def build_global_filename(var, year):
    scenario = scenario_for_year(year)
    return f"{var}_day_{MODEL}_{scenario}_{ENSEMBLE}_{GRID_LABEL}_{year}_v2.0.nc"


def build_remote_relpath(var, year):
    """相对路径：{model}/{scenario}/{ensemble}/{var}/{filename}"""
    scenario = scenario_for_year(year)
    fname = build_global_filename(var, year)
    return f"{MODEL}/{scenario}/{ENSEMBLE}/{var}/{fname}"


def build_ncss_params(year):
    return {
        "north": BBOX["north"],
        "south": BBOX["south"],
        "east":  BBOX["east"],
        "west":  BBOX["west"],
        "horizStride": 1,
        "time_start": f"{year}-01-01T12:00:00Z",
        "time_end":   f"{year}-12-31T12:00:00Z",
        "timeStride": 1,
        "accept": "netcdf",
    }


def out_path_for(var, year, var_dir):
    scenario = scenario_for_year(year)
    return os.path.join(var_dir, f"{var}_{year}_{scenario}_clipped.nc")


# ============================================================
# 模式一：NCSS 远程裁剪下载
# ============================================================

def download_via_ncss(var, year, out_path, timeout=120):
    """
    通过 NCSS 直接请求裁剪后的数据，不下载全球文件。
    成功返回 True，失败抛出异常或返回 False。
    """
    relpath = build_remote_relpath(var, year)
    url = f"{NCSS_BASE}/{relpath}"
    params = build_ncss_params(year)
    params["var"] = var

    resp = requests.get(url, params=params, timeout=timeout, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} | {resp.url}")

    # 检查返回内容是否真的是 netcdf（有时服务器返回200但是错误页面）
    content_type = resp.headers.get("Content-Type", "")
    if "netcdf" not in content_type and "octet-stream" not in content_type:
        # 读取一点内容看看是不是错误信息
        preview = next(resp.iter_content(500), b"").decode(errors="ignore")
        raise RuntimeError(f"返回非NetCDF内容（{content_type}）：{preview[:200]}")

    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"      NCSS {var}_{year}", leave=False
    ) as bar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))
    return True


# ============================================================
# 模式二：下载全球文件 + 本地裁剪（备用）
# ============================================================

def download_global_file(var, year, tmp_path, timeout=600):
    """下载未裁剪的全球 NetCDF 文件"""
    relpath = build_remote_relpath(var, year)
    url = f"{FILESERVER_BASE}/{relpath}"

    resp = requests.get(url, timeout=timeout, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} | {url}")

    total = int(resp.headers.get("content-length", 0))
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"      全球下载 {var}_{year}", leave=False
    ) as bar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))


def clip_global_file(tmp_path, out_path, var):
    """用 xarray 按 BBOX 裁剪本地全球 NetCDF 文件"""
    import xarray as xr

    ds = xr.open_dataset(tmp_path)

    # 经度可能是 0-360 或 -180-180，需要判断处理
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"

    lon_vals = ds[lon_name].values
    west, east = BBOX["west"], BBOX["east"]
    if lon_vals.max() > 180:
        # 数据用0-360表示，BBOX用-180~180或0~360均可，这里统一转0-360
        west = west % 360
        east = east % 360

    ds_clip = ds.sel(
        {lon_name: slice(west, east),
         lat_name: slice(BBOX["south"], BBOX["north"])}
    )
    # 若纬度是降序排列，slice方向要反过来
    if ds_clip[lat_name].size == 0:
        ds_clip = ds.sel(
            {lon_name: slice(west, east),
             lat_name: slice(BBOX["north"], BBOX["south"])}
        )

    ds_clip.to_netcdf(out_path)
    ds.close()
    ds_clip.close()


def download_via_global(var, year, out_path):
    """先下载全球文件，裁剪后删除原始文件"""
    tmp_path = out_path.replace("_clipped.nc", "_GLOBAL_tmp.nc")
    try:
        download_global_file(var, year, tmp_path)
        clip_global_file(tmp_path, out_path, var)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    total = len(VARIABLES) * (YEAR_END - YEAR_START + 1)
    print(f"{'='*60}")
    print(f"模式：{MODEL}  集合成员：{ENSEMBLE}  网格标签：{GRID_LABEL}")
    print(f"变量：{VARIABLES}")
    print(f"时间：{YEAR_START}—{HISTORICAL_END}（historical）+ "
          f"{HISTORICAL_END+1}—{YEAR_END}（ssp245）")
    print(f"研究区：E{BBOX['west']:.2f}°—{BBOX['east']:.2f}°，"
          f"N{BBOX['south']:.2f}°—{BBOX['north']:.2f}°")
    print(f"共 {total} 个文件，输出至：{OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    use_ncss = True          # 当前使用的模式
    consec_fail = 0          # NCSS连续失败计数

    success_list = []
    fail_list    = []

    for var in VARIABLES:
        var_dir = os.path.join(OUTPUT_DIR, var)
        os.makedirs(var_dir, exist_ok=True)
        print(f"\n▶ 变量：{var}")

        for year in range(YEAR_START, YEAR_END + 1):
            out_path = out_path_for(var, year, var_dir)

            if os.path.exists(out_path):
                success_list.append((var, year))
                continue

            scenario = scenario_for_year(year)
            print(f"  [{year}] ({scenario}) ", end="", flush=True)

            try:
                if use_ncss:
                    print("NCSS裁剪下载...", end=" ", flush=True)
                    download_via_ncss(var, year, out_path)
                    consec_fail = 0
                else:
                    print("全球下载+本地裁剪...", end=" ", flush=True)
                    download_via_global(var, year, out_path)

                print("✓")
                success_list.append((var, year))

            except Exception as e:
                print(f"✗ {e}")
                fail_list.append((var, year, str(e)))

                if use_ncss:
                    consec_fail += 1
                    if consec_fail >= MAX_CONSEC_FAIL:
                        print(f"\n  [提示] NCSS连续失败{MAX_CONSEC_FAIL}次，"
                              f"自动切换为【全球下载+本地裁剪】模式\n")
                        use_ncss = False
                        consec_fail = 0

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成：成功 {len(success_list)} 个，失败 {len(fail_list)} 个")

    if fail_list:
        log_path = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
        with open(log_path, "w") as f:
            for var, year, err in fail_list:
                f.write(f"{var}\t{year}\t{err}\n")
        print(f"失败列表已保存：{log_path}")
        print("（重新运行脚本，已存在文件会自动跳过，仅补充下载失败的部分）")


if __name__ == "__main__":
    main()
