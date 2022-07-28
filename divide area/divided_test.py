import os
from datetime import datetime, timedelta
import netCDF4
import numpy as np

def get_time(delta):
    start = datetime(1900, 1, 1)
    delta = timedelta(hours=int(delta))
    return start + delta


if __name__ == '__main__':
    for root, _, files in os.walk(r"divided_nc"):
        for file in files:
            if file.endswith("monthly.nc"):
                path = os.path.join(root, file)
                print(file)
                nc = netCDF4.Dataset(path)
                time = nc["time"][:]
                PRAVG = nc["PRAVG"][:]
                times = []
                for ti, hour in enumerate(time):
                    print(get_time(hour))
                    print("Mean:", np.mean(PRAVG[ti]))
                    print("Median:", np.median(PRAVG[ti]))
                    print("")
