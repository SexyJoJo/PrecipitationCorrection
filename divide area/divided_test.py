import os

import netCDF4

if __name__ == '__main__':
    for root, _, files in os.walk(r"divided_nc"):
        for file in files:
            if file.endswith(".nc"):
                path = os.path.join(root, file)
                print(file)
                nc = netCDF4.Dataset(path)
                XLONG = nc["XLONG"][:]
                XLAT = nc["XLAT"][:]
                time = nc["time"][:]
                PRAVG = nc["PRAVG"][:]
                print(PRAVG)