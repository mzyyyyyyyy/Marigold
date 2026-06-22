"""
CHELSA V2.1 逐月气候数据下载脚本
研究区：云南、湖南、广西、贵州四省
时间范围：1986—2021年
变量：tas（均温）、pr（降水）、rsds（太阳辐射）、spei12（SPEI）

依赖安装：
    pip install requests rasterio numpy tqdm shapely
"""

import os
import requests
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
from rasterio.env import Env
from shapely.geometry import box, mapping

# ============================================================
# 配置参数（按需修改）
# ============================================================

BBOX = {
    "west":  96.9999998617999921,   # 云南西部
    "south": 19.9997332551999989,   # 广西南部
    "east":  115.0000831849999940,  # 湖南东部
    "north": 31.0002333326000006    # 湖南北部
}

YEAR_START = 1986
YEAR_END   = 2021

# 只下载指定月份（可改为多个月，如 [6, 7, 8] 表示夏季三个月）
TARGET_MONTHS = [12]

VARIABLES = [
    "tas",     # 月均温（下载后单位K，自动转为℃）
    "pr",      # 月降水量（mm/month）
    "spei12",  # SPEI-12（spei12只到2018年）
    "tasmin",  # 月均最低温（K -> ℃）
    "pet",       # 潜在蒸散发（mm/month）
    "vpd"  
]

OUTPUT_DIR = "./chelsa_data_w"

# CHELSA V2.1 文件 URL 模板
# 不同变量服务器上的 URL 结构不同：
#
#   A类 tas, pr
#       .../monthly/{var}/CHELSA_{var}_{month}_{year}_V.2.1.tif
#       文件名：月_年，无年份子目录
#
#   B类 rsds, pet 等
#       .../monthly/{var}/CHELSA_{var}_{year}_{month}_V.2.1.tif
#       文件名：年_月，无年份子目录
#
#   C类 spei12, spi12 等
#       .../monthly/{var}/{year}/CHELSA_{var}_{month}_{year}_V.2.1.tif
#       文件名：月_年，有年份子目录

# 不同变量的域名、路径结构、文件名格式均可能不同，按实际 URL 分组：
#
#   A类 tas, pr          域名 zhdk，无年份子目录，文件名 月_年
#   B类 rsds 等          域名 zhdk，无年份子目录，文件名 年_月
#   C类 spei12, spi12    域名 unil，有年份子目录，文件名 月_年
#   D类 pet              域名 zhdk，有年份子目录，文件名 月_年

DOMAIN_ZHDK = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/monthly"
DOMAIN_UNIL = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/monthly"

URL_TYPE_A = {"tas", "pr"}        # zhdk，无年份子目录，月_年
URL_TYPE_C = {"spei12", "spi12", "tasmin", "tasmax", "vpd", "clt", "cmi", "hurs", "pet", "sfcWind"}  # unil，有年份子目录，月_年
URL_TYPE_D = {"pet"}              # zhdk，有年份子目录，月_年
# 其余变量默认 B 类：zhdk，无年份子目录，年_月

# ============================================================
# 工具函数
# ============================================================

def build_url(var, year, month):
    if var in URL_TYPE_A:
        # zhdk .../monthly/{var}/CHELSA_{var}_{月}_{年}_V.2.1.tif
        return f"{DOMAIN_ZHDK}/{var}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    elif var in URL_TYPE_C:
        # unil .../monthly/{var}/{年}/CHELSA_{var}_{月}_{年}_V.2.1.tif
        return f"{DOMAIN_UNIL}/{var}/{year}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    elif var in URL_TYPE_D:
        # zhdk .../monthly/{var}/{年}/CHELSA_{var}_{月}_{年}_V.2.1.tif
        return f"{DOMAIN_ZHDK}/{var}/{year}/CHELSA_{var}_{month:02d}_{year}_V.2.1.tif"
    else:
        # zhdk .../monthly/{var}/CHELSA_{var}_{年}_{月}_V.2.1.tif
        return f"{DOMAIN_ZHDK}/{var}/CHELSA_{var}_{year}_{month:02d}_V.2.1.tif"


def clip_raster(src, bbox):
    """用边界框裁剪已打开的 rasterio 数据集，返回 (data, transform, meta)"""
    geom = box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    out_image, out_transform = mask(src, [mapping(geom)], crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver":    "GTiff",
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform,
        "compress":  "lzw",
    })
    return out_image, out_meta


def apply_unit_conversion(data, var, nodata=None):
    """tas: K -> ℃；其余变量不变"""
    if var != "tas":
        return data
    result = data.astype("float32")
    if nodata is not None:
        valid = result != nodata
        result[valid] -= 273.15
    else:
        result -= 273.15
    return result


def save_raster(path, data, meta):
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)


# ============================================================
# 方式一：远程直接裁剪（推荐，无需下载全球文件）
# ============================================================

def process_remote(var, year, month, out_path):
    """
    通过 /vsicurl/ 直接读取远程 COG 文件并裁剪，
    只传输研究区对应的数据块（约3-5MB），不下载全球文件。
    """
    url = build_url(var, year, month)
    vsicurl_path = f"/vsicurl/{url}"

    gdal_env = dict(
        GDAL_HTTP_UNSAFESSL="YES",          # 跳过SSL自签名证书问题
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_USE_HEAD="NO",
    )

    with Env(**gdal_env):
        with rasterio.open(vsicurl_path) as src:
            out_image, out_meta = clip_raster(src, BBOX)
            nodata = src.nodata

    out_image = apply_unit_conversion(out_image, var, nodata)
    save_raster(out_path, out_image, out_meta)


# ============================================================
# 方式二：先下载全局文件再裁剪（备用方案）
# ============================================================

def process_download(var, year, month, out_path):
    """
    先将全球 GeoTIFF (~300MB) 下载到临时文件，
    裁剪后删除原始文件，仅保留研究区文件。
    """
    url = build_url(var, year, month)
    tmp_path = out_path.replace(".tif", "_GLOBAL_tmp.tif")

    # 下载
    try:
        resp = requests.get(url, stream=True, timeout=300, verify=False)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"      下载 {os.path.basename(tmp_path)}", leave=False
        ) as bar:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"下载失败: {e}")

    # 裁剪
    try:
        with rasterio.open(tmp_path) as src:
            out_image, out_meta = clip_raster(src, BBOX)
            nodata = src.nodata
        out_image = apply_unit_conversion(out_image, var, nodata)
        save_raster(out_path, out_image, out_meta)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)   # 无论成功失败都删除全球文件


# ============================================================
# 主流程
# ============================================================

def main():
    # 创建目录
    for var in VARIABLES:
        os.makedirs(os.path.join(OUTPUT_DIR, var), exist_ok=True)

    total = len(VARIABLES) * (YEAR_END - YEAR_START + 1) * len(TARGET_MONTHS)
    print(f"\n{'='*60}")
    print(f"研究区：E{BBOX['west']}°—{BBOX['east']}°，N{BBOX['south']}°—{BBOX['north']}°")
    print(f"时间：{YEAR_START}—{YEAR_END}年，月份：{TARGET_MONTHS}")
    print(f"变量：{VARIABLES}")
    print(f"共 {total} 个文件，输出至：{OUTPUT_DIR}/")
    print(f"{'='*60}\n")

    # ---- 选择处理模式 ----
    # 优先尝试远程裁剪；若连续失败则自动切换为下载模式
    use_remote = True
    remote_fail_count = 0
    REMOTE_FAIL_THRESHOLD = 3   # 连续失败超过此数切换为下载模式

    success_list = []
    fail_list    = []

    for var in VARIABLES:
        print(f"\n▶ 变量：{var}")
        # spei12 只有到 2018 年
        year_end_var = min(YEAR_END, 2018) if var == "spei12" else YEAR_END

        for year in range(YEAR_START, year_end_var + 1):
            for month in TARGET_MONTHS:
                out_path = os.path.join(
                    OUTPUT_DIR, var,
                    f"CHELSA_{var}_{year}_{month:02d}_clipped.tif"
                )

                if os.path.exists(out_path):
                    success_list.append((var, year, month))
                    continue

                print(f"  [{year}-{month:02d}] ", end="", flush=True)

                try:
                    if use_remote:
                        print("远程裁剪...", end=" ", flush=True)
                        process_remote(var, year, month, out_path)
                        remote_fail_count = 0
                    else:
                        print("下载裁剪...", end=" ", flush=True)
                        process_download(var, year, month, out_path)

                    print("✓")
                    success_list.append((var, year, month))

                except Exception as e:
                    print(f"✗ {e}")
                    fail_list.append((var, year, month))

                    if use_remote:
                        remote_fail_count += 1
                        if remote_fail_count >= REMOTE_FAIL_THRESHOLD:
                            print(f"\n  [提示] 远程裁剪连续失败 {REMOTE_FAIL_THRESHOLD} 次，"
                                  f"自动切换为【下载裁剪】模式\n")
                            use_remote = False
                            remote_fail_count = 0

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成：成功 {len(success_list)} 个，失败 {len(fail_list)} 个")

    if fail_list:
        log = os.path.join(OUTPUT_DIR, "failed.txt")
        with open(log, "w") as f:
            for var, year, month in fail_list:
                f.write(f"{var}\t{year}\t{month:02d}\t{build_url(var,year,month)}\n")
        print(f"失败列表已保存：{log}（重新运行脚本可自动补充下载）")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
