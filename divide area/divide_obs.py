import netCDF4
import os

if __name__ == '__main__':
    # 修改数据路径与保存路径
    data_path = r"E:\Data\PrecipitationCorrection\OBS"
    save_path = r"./divided obs"

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".nc"):
                path = os.path.join(root, file)
                print(file)
                nc = netCDF4.Dataset(path)

                print(nc.dimensions)
                prec = nc.variables["prec"][:]
                lon2d = nc.variables["lon2d"][:]
                lat2d = nc.variables["lat2d"][:]

                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # 流域经纬度划分
                # 金沙江流域划分
                JSJ_XLONG = lon2d[53:96, 48:87]
                JSJ_XLAT = lat2d[53:96, 48:87]
                JSJ_PRAVG = prec[53:96, 48:87]
                JSJ_nc = netCDF4.Dataset(os.path.join(save_path, 'JSJ_' + file), 'w', format='NETCDF4')
                JSJ_nc.createDimension('x', JSJ_XLONG.shape[0])
                JSJ_nc.createDimension('y', JSJ_XLONG.shape[1])
                JSJ_nc.createVariable('lon2d', "f", ("x", "y"))
                JSJ_nc.createVariable('lat2d', "f", ("x", "y"))
                JSJ_nc.createVariable('prec', "f", ("x", "y"))
                JSJ_nc.variables["lon2d"][:] = JSJ_XLONG
                JSJ_nc.variables["lat2d"][:] = JSJ_XLAT
                JSJ_nc.variables["prec"][:] = JSJ_PRAVG

                # 长江上游流域划分
                CJSY_XLONG = lon2d[46:88, 80:116]
                CJSY_XLAT = lat2d[46:88, 80:116]
                CJSY_PRAVG = prec[46:88, 80:116]
                CJSY_nc = netCDF4.Dataset(os.path.join(save_path, 'CJSY_' + file), 'w', format='NETCDF4')
                CJSY_nc.createDimension('x', CJSY_XLONG.shape[0])
                CJSY_nc.createDimension('y', CJSY_XLONG.shape[1])
                CJSY_nc.createVariable('lon2d', "f", ("x", "y"))
                CJSY_nc.createVariable('lat2d', "f", ("x", "y"))
                CJSY_nc.createVariable('prec', "f", ("x", "y"))
                CJSY_nc.variables["lon2d"][:] = CJSY_XLONG
                CJSY_nc.variables["lat2d"][:] = CJSY_XLAT
                CJSY_nc.variables["prec"][:] = CJSY_PRAVG

                CJZXY_XLONG = lon2d[46:88, 116:158]
                CJZXY_XLAT = lat2d[46:88, 116:158]
                CJZXY_PRAVG = prec[46:88, 116:158]
                CJZXY_nc = netCDF4.Dataset(os.path.join(save_path, 'CJZXY_' + file), 'w', format='NETCDF4')
                CJZXY_nc.createDimension('x', CJZXY_XLONG.shape[0])
                CJZXY_nc.createDimension('y', CJZXY_XLONG.shape[1])
                CJZXY_nc.createVariable('lon2d', "f", ("x", "y"))
                CJZXY_nc.createVariable('lat2d', "f", ("x", "y"))
                CJZXY_nc.createVariable('prec', "f", ("x", "y"))
                CJZXY_nc.variables["lon2d"][:] = CJZXY_XLONG
                CJZXY_nc.variables["lat2d"][:] = CJZXY_XLAT
                CJZXY_nc.variables["prec"][:] = CJZXY_PRAVG

