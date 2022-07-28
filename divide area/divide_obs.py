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
                lat2d = nc.variables["lat2d"][:]
                lon2d = nc.variables["lon2d"][:]

                print(prec)

