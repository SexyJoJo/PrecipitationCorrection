import os
import netCDF4


if __name__ == '__main__':
    # 修改数据路径与保存路径
    data_path = r"D:\Data\PrecipitationCorrection\CASE"
    save_base_path = r"./divided case"

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith("monthly.nc"):
                path = os.path.join(root, file)
                print(file)
                nc = netCDF4.Dataset(path)
                XLONG = nc["XLONG"][:]
                XLAT = nc["XLAT"][:]
                time = nc["time"][:]
                PRAVG = nc["PRAVG"][:]  # 提取单个值：PRAVG[时间][行][列]
                # 金沙江流域（左矩形）：经度范围：90-100 纬度范围：36-26 rowmin:53 rowmax:95 colmin:48 colmax:86
                # 长江上游（中矩形）：经度范围：100-110 纬度范围：24-35 rowmin:46 rowmax:87 colmin:80 colmax:115
                # 长江中下游（右矩形）：经度范围：110-122 纬度范围：24-35 rowmin:46 rowmax:87 colmin:116 colmax:157
                dirs = root.split("\\")
                save_path = os.path.join(save_base_path, dirs[-3], dirs[-2], dirs[-1], "ChangJiang")
                os.makedirs(save_path, exist_ok=True)

                # 流域经纬度划分
                # 金沙江流域划分
                JSJ_XLONG = XLONG[53:96, 48:87]
                JSJ_XLAT = XLAT[53:96, 48:87]
                JSJ_PRAVG = PRAVG[:, 53:96, 48:87]
                JSJ_nc = netCDF4.Dataset(os.path.join(save_path, 'JSJ_' + file), 'w', format='NETCDF4')
                JSJ_nc.createDimension('time', time.shape[0])
                JSJ_nc.createDimension('x', JSJ_XLONG.shape[0])
                JSJ_nc.createDimension('y', JSJ_XLONG.shape[1])
                JSJ_nc.createVariable('time', "f", ("time", ))
                JSJ_nc.createVariable('XLONG', "f", ("x", "y"))
                JSJ_nc.createVariable('XLAT', "f", ("x", "y"))
                JSJ_nc.createVariable('PRAVG', "f", ("time", "x", "y"))
                JSJ_nc.variables["XLONG"][:] = JSJ_XLONG
                JSJ_nc.variables["XLAT"][:] = JSJ_XLONG
                JSJ_nc.variables["time"][:] = time
                JSJ_nc.variables["PRAVG"][:] = JSJ_PRAVG

                # 长江上游流域划分
                CJSY_XLONG = XLONG[46:88, 80:116]
                CJSY_XLAT = XLAT[46:88, 80:116]
                CJSY_PRAVG = PRAVG[:, 46:88, 80:116]
                CJSY_nc = netCDF4.Dataset(os.path.join(save_path, 'CJSY_' + file), 'w', format='NETCDF4')
                CJSY_nc.createDimension('time', time.shape[0])
                CJSY_nc.createDimension('x', CJSY_XLONG.shape[0])
                CJSY_nc.createDimension('y', CJSY_XLONG.shape[1])
                CJSY_nc.createVariable('time', "f", ("time",))
                CJSY_nc.createVariable('XLONG', "f", ("x", "y"))
                CJSY_nc.createVariable('XLAT', "f", ("x", "y"))
                CJSY_nc.createVariable('PRAVG', "f", ("time", "x", "y"))
                CJSY_nc.variables["XLONG"][:] = CJSY_XLONG
                CJSY_nc.variables["XLAT"][:] = CJSY_XLAT
                CJSY_nc.variables["time"][:] = time
                CJSY_nc.variables["PRAVG"][:] = CJSY_PRAVG

                CJZXY_XLONG = XLONG[46:88, 116:158]
                CJZXY_XLAT = XLAT[46:88, 116:158]
                CJZXY_PRAVG = PRAVG[:, 46:88, 116:158]
                CJZXY_nc = netCDF4.Dataset(os.path.join(save_path, 'CJZXY_' + file), 'w', format='NETCDF4')
                CJZXY_nc.createDimension('time', time.shape[0])
                CJZXY_nc.createDimension('x', CJZXY_XLONG.shape[0])
                CJZXY_nc.createDimension('y', CJZXY_XLONG.shape[1])
                CJZXY_nc.createVariable('time', "f", ("time",))
                CJZXY_nc.createVariable('XLONG', "f", ("x", "y"))
                CJZXY_nc.createVariable('XLAT', "f", ("x", "y"))
                CJZXY_nc.createVariable('PRAVG', "f", ("time", "x", "y"))
                CJZXY_nc.variables["XLONG"][:] = CJZXY_XLONG
                CJZXY_nc.variables["XLAT"][:] = CJZXY_XLAT
                CJZXY_nc.variables["time"][:] = time
                CJZXY_nc.variables["PRAVG"][:] = CJZXY_PRAVG
