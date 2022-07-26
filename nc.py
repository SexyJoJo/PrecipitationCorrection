import netCDF4
import numpy as np


if __name__ == '__main__':
    path = r"D:\Data\PrecipitationCorrection\CASE1\TIME00\PRAVG_1991030200c01_1991_monthly.nc"
    nc = netCDF4.Dataset(path)
    print(nc.variables.keys())
    XLONG = np.asarray(nc["XLONG"][:])
    XLAT = np.asarray(nc["XLAT"][:])
    time = nc["time"][:]
    PRAVG = nc["PRAVG"][:]  # 提取单个值：PRAVG[时间][行][列]
    print(XLONG)
    print(len(PRAVG))

    # TP（左矩形）：经度范围：90-100 纬度范围：36-26 rowmin:53 rowmax:95 colmin:48 colmax:86
    # SWCH（中矩形）：经度范围：100-110 纬度范围：24-35
    # YHRB（右矩形）：经度范围：110-122 纬度范围：24-35
