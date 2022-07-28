import netCDF4
import os

if __name__ == '__main__':
    # 修改数据路径与保存路径
    data_path = r"E:\Data\PrecipitationCorrection\OBS"
    save_path = r"./divided obs"

    nc = netCDF4.Dataset(r"E:\Data\PrecipitationCorrection\CN_OBS_PRAVG_monthly.nc")
    print(nc.variables.keys())
    a = nc.variables['PRAVG'][-2][-1][:]
    print(r"E:\Data\PrecipitationCorrection\CN_OBS_PRAVG_monthly.nc")
    print(nc.variables['PRAVG'][-2][-1][88][30], "\n")

    nc = netCDF4.Dataset(r"E:\Data\PrecipitationCorrection\OBS\obs_prec_rcm_201412.nc")
    print(nc.variables.keys())
    b = nc.variables['prec'][:]
    print(r"E:\Data\PrecipitationCorrection\OBS\obs_prec_rcm_201412.nc")
    print(nc.variables['prec'][88][30])
    # for root, _, files in os.walk(data_path):
    #     for file in files:
    #         if file.endswith(".nc"):
    #             path = os.path.join(root, file)
    #             print(file)
    #             nc = netCDF4.Dataset(path)
    #
    #             print(nc.dimensions)
    #             prec = nc.variables["prec"][:]
    #             lat2d = nc.variables["lat2d"][:]
    #             lon2d = nc.variables["lon2d"][:]
    #
    #             print(prec)

