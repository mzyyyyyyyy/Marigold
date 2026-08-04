"""
CHELSA V2.1 年均气候数据下载脚本
==================================
流程：
  1. 下载每年 1-12 月的逐月 tif（下载到临时目录）
  2. 对 12 个月求均值，生成年均值 tif（保存到输出目录）
  3. 删除临时月份文件，节省磁盘空间

输出文件命名：CHELSA_{var}_{year}_annual_clipped.tif
输出目录：chelsa_data_annual/{var}/

变量：hurs, pr, rsds, sfcWind, tas, tasmax, tasmin
时间：1986-2018年

依赖安装：
  pip install requests rasterio numpy tqdm shapely
"""

import os
import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.env import Env
from shapely.geometry import box, mapping
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 配置参数
# ============================================================


BBOX = {
    "west":  96.9999998617999921,   # 云南西部
    "south": 19.9997332551999989,   # 广西南部
    "east":  115.0000831849999940,  # 湖南东部
    "north": 31.0002333326000006    # 湖南北部
}

YEAR_START = 1986
YEAR_END   = 2018

VARIABLES = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]

OUTPUT_DIR = "/mnt/data/china_project/chelsa_data_annual"   # 年均值输出目录
TEMP_DIR   = "/mnt/data/china_project/chelsa_temp_monthly"  # 临时月份文件目录

# URL 分类（与原下载脚本一致）
DOMAIN_ZHDK = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/monthly"
DOMAIN_UNIL = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/monthly"

URL_TYPE_A = {"tas", "pr"}        # zhdk，月_年，无年份子目录
URL_TYPE_C = {"spei12", "spi12", "hurs", "sfcWind", "tasmax", "tasmin"}  # unil，月_年，有年份子目录
URL_TYPE_D = {"pet"}              # zhdk，月_年，有年份子目录
# 其余（hurs, rsds, sfcWind, tasmax, tasmin）为 B 类：zhdk，年_月，无年份子目录

REMOTE_FAIL_THRESHOLD = 3

# ============================================================
# URL 构建
# ============================================================

def build_url(var, year, month):
    if var in URL_TYPE_A:
        return f"{DOMAIN_ZHDK}/{var}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    elif var in URL_TYPE_C:
        return f"{DOMAIN_UNIL}/{var}/{year}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    elif var in URL_TYPE_D:
        return f"{DOMAIN_ZHDK}/{var}/{year}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    else:
        # B 类：hurs, rsds, sfcWind, tasmax, tasmin
        return f"{DOMAIN_ZHDK}/{var}/CHELSA_{var}_{year}_{month:02d}_V.2.1.tif"


# ============================================================
# 下载单个月份文件
# ============================================================

def download_month_remote(var, year, month, out_path):
    """远程裁剪下载（优先）"""
    url          = build_url(var, year, month)
    vsicurl_path = f"/vsicurl/{url}"
    gdal_env     = dict(
        GDAL_HTTP_UNSAFESSL="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_USE_HEAD="NO",
    )
    geom = box(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])

    with Env(**gdal_env):
        with rasterio.open(vsicurl_path) as src:
            out_image, out_transform = mask(src, [mapping(geom)], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver":    "GTiff",
                "height":    out_image.shape[1],
                "width":     out_image.shape[2],
                "transform": out_transform,
                "compress":  "lzw",
                "dtype":     "float32",
            })
            nodata = src.nodata

    # tas 单位转换 K -> ℃
    if var in {"tas", "tasmax", "tasmin"}:
        out_image = out_image.astype("float32")
        if nodata is not None:
            out_image[out_image != nodata] -= 273.15
        else:
            out_image -= 273.15

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)


def download_month_global(var, year, month, out_path):
    """下载全球文件后本地裁剪（备用）"""
    url      = build_url(var, year, month)
    tmp_path = out_path.replace(".tif", "_GLOBAL_tmp.tif")
    geom     = box(BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"])

    try:
        resp = requests.get(url, stream=True, timeout=300, verify=False)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"      下载 {var}_{year}_{month:02d}", leave=False
        ) as bar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))

        with rasterio.open(tmp_path) as src:
            out_image, out_transform = mask(src, [mapping(geom)], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver":    "GTiff",
                "height":    out_image.shape[1],
                "width":     out_image.shape[2],
                "transform": out_transform,
                "compress":  "lzw",
                "dtype":     "float32",
            })
            nodata = src.nodata

        if var in {"tas", "tasmax", "tasmin"}:
            out_image = out_image.astype("float32")
            if nodata is not None:
                out_image[out_image != nodata] -= 273.15
            else:
                out_image -= 273.15

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# 计算年均值
# ============================================================

def compute_annual_mean(var, year, monthly_paths, output_path):
    """
    读取 12 个月的 tif，求均值，保存年均值 tif。
    pr（降水）求年总量（累加），其余变量求年均值（平均）。
    """
    arrays = []
    meta   = None

    for path in monthly_paths:
        if not os.path.exists(path):
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arrays.append(arr)
            if meta is None:
                meta = src.profile.copy()

    if not arrays:
        print(f"    [警告] {var} {year} 没有任何月份数据，跳过")
        return False

    stack = np.stack(arrays, axis=0)   # (n_months, H, W)

    # pr 求年总量（mm/year），其余求年均值
    if var == "pr":
        annual = np.nansum(stack, axis=0)
    else:
        annual = np.nanmean(stack, axis=0)

    meta.update({
        "dtype":   "float32",
        "nodata":  np.nan,
        "compress": "lzw",
        "count":   1,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(annual, 1)

    return True


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,   exist_ok=True)

    total = len(VARIABLES) * (YEAR_END - YEAR_START + 1)
    print(f"{'='*60}")
    print(f"变量：{VARIABLES}")
    print(f"时间：{YEAR_START}—{YEAR_END}年，逐月下载后求年均值")
    print(f"输出：{OUTPUT_DIR}/")
    print(f"共 {total} 个年均值文件")
    print(f"{'='*60}\n")

    use_remote        = True
    remote_fail_count = 0
    success_list, fail_list = [], []

    for var in VARIABLES:
        var_out_dir  = os.path.join(OUTPUT_DIR, var)
        var_temp_dir = os.path.join(TEMP_DIR,   var)
        os.makedirs(var_out_dir,  exist_ok=True)
        os.makedirs(var_temp_dir, exist_ok=True)
        print(f"\n▶ 变量：{var}")

        for year in tqdm(range(YEAR_START, YEAR_END + 1),
                         desc=f"  {var}", leave=False):

            annual_path = os.path.join(var_out_dir,
                                       f"CHELSA_{var}_{year}_annual_clipped.tif")
            if os.path.exists(annual_path):
                success_list.append((var, year))
                continue

            # ---- 下载 12 个月 ----
            monthly_paths = []
            month_ok      = True

            for month in range(1, 13):
                month_path = os.path.join(
                    var_temp_dir,
                    f"CHELSA_{var}_{year}_{month:02d}_clipped.tif"
                )
                monthly_paths.append(month_path)

                if os.path.exists(month_path):
                    continue

                try:
                    if use_remote:
                        download_month_remote(var, year, month, month_path)
                        remote_fail_count = 0
                    else:
                        download_month_global(var, year, month, month_path)

                except Exception as e:
                    print(f"\n  [{var} {year}-{month:02d}] ✗ {e}")
                    month_ok = False

                    if use_remote:
                        remote_fail_count += 1
                        if remote_fail_count >= REMOTE_FAIL_THRESHOLD:
                            print(f"  [提示] 远程连续失败，切换为全局下载模式")
                            use_remote        = False
                            remote_fail_count = 0

            # ---- 求年均值 ----
            ok = compute_annual_mean(var, year, monthly_paths, annual_path)

            if ok:
                success_list.append((var, year))
                # 删除临时月份文件，节省磁盘
                for p in monthly_paths:
                    if os.path.exists(p):
                        os.remove(p)
            else:
                fail_list.append((var, year))

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成：成功 {len(success_list)} 个，失败 {len(fail_list)} 个")

    if fail_list:
        log = os.path.join(OUTPUT_DIR, "failed.txt")
        with open(log, "w") as f:
            for var, year in fail_list:
                f.write(f"{var}\t{year}\n")
        print(f"失败列表：{log}（重新运行可自动补充）")


if __name__ == "__main__":
    main()