"""
NASA NEX-GDDP-CMIP6 下载脚本（GFDL-ESM4，多情景版本）
======================================================
下载范围：1986—2100年，云南/湖南/广西/贵州研究区
变量：tas, tasmin, tasmax, pr, rsds, hurs, sfcWind
情景：ssp126 和 ssp585（historical 段两个情景共用，只下载一次）

文件命名规则（所有文件放在同一变量子文件夹下）：
  historical : {var}_{year}_historical_clipped.nc
  ssp126     : {var}_{year}_ssp126_clipped.nc
  ssp585     : {var}_{year}_ssp585_clipped.nc

两种下载模式（自动切换）：
  模式一【优先】：NCSS远程裁剪，只下载研究区数据
  模式二【备用】：下载全球NetCDF文件，本地裁剪后删除原始文件

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

MODEL      = "GFDL-ESM4"
ENSEMBLE   = "r1i1p1f1"
GRID_LABEL = "gr1"

VARIABLES = ["tas", "tasmin", "tasmax", "pr", "rsds", "hurs", "sfcWind"]

BBOX = {
    "west":  96.9999998617999921,
    "south": 19.9997332551999989,
    "east":  115.0000831849999940,
    "north": 31.0002333326000006,
}

YEAR_START     = 1986
YEAR_END       = 2100
HISTORICAL_END = 2014   # 2014及以前用historical，之后用SSP情景

# ---- 需要下载的未来情景 ----
# historical 段两个情景共用，只下载一次（已存在则跳过）
FUTURE_SCENARIOS = ["ssp126", "ssp585"]

OUTPUT_DIR = "/mnt/data/china_project/cmip6_data"
TEMP_DIR   = "/mnt/data/china_project/cmip6_temp_global"

THREDDS_BASE    = "https://ds.nccs.nasa.gov/thredds"
NCSS_BASE       = f"{THREDDS_BASE}/ncss/grid/AMES/NEX/GDDP-CMIP6"
FILESERVER_BASE = f"{THREDDS_BASE}/fileServer/AMES/NEX/GDDP-CMIP6"

MAX_CONSEC_FAIL = 3


# ============================================================
# 工具函数
# ============================================================

def build_global_filename(var, year, scenario):
    return f"{var}_day_{MODEL}_{scenario}_{ENSEMBLE}_{GRID_LABEL}_{year}_v2.0.nc"


def build_remote_relpath(var, year, scenario):
    fname = build_global_filename(var, year, scenario)
    return f"{MODEL}/{scenario}/{ENSEMBLE}/{var}/{fname}"


def build_ncss_params(var, year):
    return {
        "var":         var,
        "north":       BBOX["north"],
        "south":       BBOX["south"],
        "east":        BBOX["east"],
        "west":        BBOX["west"],
        "horizStride": 1,
        "time_start":  f"{year}-01-01T12:00:00Z",
        "time_end":    f"{year}-12-31T12:00:00Z",
        "timeStride":  1,
        "accept":      "netcdf",
    }


def local_path(var, year, scenario):
    """本地保存路径，所有情景放在同一变量子文件夹下，文件名带情景标识"""
    var_dir = os.path.join(OUTPUT_DIR, var)
    return os.path.join(var_dir, f"{var}_{year}_{scenario}_clipped.nc")


# ============================================================
# 模式一：NCSS 远程裁剪下载
# ============================================================

def download_via_ncss(var, year, scenario, out_path, timeout=120):
    relpath = build_remote_relpath(var, year, scenario)
    url     = f"{NCSS_BASE}/{relpath}"
    params  = build_ncss_params(var, year)

    resp = requests.get(url, params=params, timeout=timeout, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} | {resp.url}")

    content_type = resp.headers.get("Content-Type", "")
    if "netcdf" not in content_type and "octet-stream" not in content_type:
        preview = next(resp.iter_content(500), b"").decode(errors="ignore")
        raise RuntimeError(f"非NetCDF响应（{content_type}）：{preview[:200]}")

    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=f"      NCSS {var}_{year}_{scenario}", leave=False
    ) as bar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))


# ============================================================
# 模式二：下载全球文件 + 本地裁剪（备用）
# ============================================================

def download_via_global(var, year, scenario, out_path, timeout=600):
    import xarray as xr

    relpath  = build_remote_relpath(var, year, scenario)
    url      = f"{FILESERVER_BASE}/{relpath}"
    tmp_path = out_path.replace("_clipped.nc", "_GLOBAL_tmp.nc")

    try:
        # 下载全球文件
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} | {url}")

        total = int(resp.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"      全球下载 {var}_{year}_{scenario}", leave=False
        ) as bar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))

        # 本地裁剪
        ds = xr.open_dataset(tmp_path)
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        lat_name = "lat" if "lat" in ds.coords else "latitude"

        west, east = BBOX["west"], BBOX["east"]
        if ds[lon_name].values.max() > 180:
            west = west % 360
            east = east % 360

        ds_clip = ds.sel({lon_name: slice(west, east),
                          lat_name: slice(BBOX["south"], BBOX["north"])})
        if ds_clip[lat_name].size == 0:
            ds_clip = ds.sel({lon_name: slice(west, east),
                              lat_name: slice(BBOX["north"], BBOX["south"])})

        ds_clip.to_netcdf(out_path)
        ds.close()
        ds_clip.close()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 下载单个文件（含自动切换逻辑）
# ============================================================

def download_one(var, year, scenario, use_ncss, consec_fail):
    """
    下载单个文件，返回 (成功与否, 更新后的use_ncss, 更新后的consec_fail)
    """
    out_path = local_path(var, year, scenario)

    if os.path.exists(out_path):
        return True, use_ncss, consec_fail

    print(f"  [{year}] ({scenario}) ", end="", flush=True)

    try:
        if use_ncss:
            print("NCSS裁剪...", end=" ", flush=True)
            download_via_ncss(var, year, scenario, out_path)
            consec_fail = 0
        else:
            print("全球下载+裁剪...", end=" ", flush=True)
            download_via_global(var, year, scenario, out_path)

        print("✓")
        return True, use_ncss, consec_fail

    except Exception as e:
        print(f"✗ {e}")
        if use_ncss:
            consec_fail += 1
            if consec_fail >= MAX_CONSEC_FAIL:
                print(f"\n  [提示] NCSS连续失败{MAX_CONSEC_FAIL}次，切换为全球下载模式\n")
                use_ncss    = False
                consec_fail = 0
        return False, use_ncss, consec_fail


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,   exist_ok=True)

    # historical 段文件数（两个情景共用，只下载一次）
    n_hist   = len(VARIABLES) * (HISTORICAL_END - YEAR_START + 1)
    # 未来情景文件数
    n_future = len(VARIABLES) * (YEAR_END - HISTORICAL_END) * len(FUTURE_SCENARIOS)
    total    = n_hist + n_future

    print(f"{'='*60}")
    print(f"模式：{MODEL}  集合成员：{ENSEMBLE}")
    print(f"变量：{VARIABLES}")
    print(f"历史段：{YEAR_START}—{HISTORICAL_END}（historical，共用）")
    print(f"未来段：{HISTORICAL_END+1}—{YEAR_END}（{FUTURE_SCENARIOS}，分别下载）")
    print(f"共 {total} 个文件，输出至：{OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    use_ncss    = True
    consec_fail = 0
    success_list, fail_list = [], []

    for var in VARIABLES:
        os.makedirs(os.path.join(OUTPUT_DIR, var), exist_ok=True)
        print(f"\n▶ 变量：{var}")

        # ---- 历史段：只下载一次，两个情景共用 ----
        print(f"  -- historical 段（{YEAR_START}-{HISTORICAL_END}）--")
        for year in range(YEAR_START, HISTORICAL_END + 1):
            ok, use_ncss, consec_fail = download_one(
                var, year, "historical", use_ncss, consec_fail)
            if ok:
                success_list.append((var, year, "historical"))
            else:
                fail_list.append((var, year, "historical"))

        # ---- 未来段：每个情景分别下载 ----
        for scenario in FUTURE_SCENARIOS:
            print(f"\n  -- {scenario} 段（{HISTORICAL_END+1}-{YEAR_END}）--")
            for year in range(HISTORICAL_END + 1, YEAR_END + 1):
                ok, use_ncss, consec_fail = download_one(
                    var, year, scenario, use_ncss, consec_fail)
                if ok:
                    success_list.append((var, year, scenario))
                else:
                    fail_list.append((var, year, scenario))

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成：成功 {len(success_list)} 个，失败 {len(fail_list)} 个")

    if fail_list:
        log_path = os.path.join(OUTPUT_DIR, "failed_downloads.txt")
        with open(log_path, "w") as f:
            for var, year, scenario in fail_list:
                f.write(f"{var}\t{year}\t{scenario}\n")
        print(f"失败列表已保存：{log_path}（重新运行可自动补充下载）")


if __name__ == "__main__":
    main()