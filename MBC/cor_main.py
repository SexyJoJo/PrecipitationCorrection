import os
import netCDF4 as nc
import numpy as np
import MBC
import matplotlib.pyplot as plt

root_path = "./"


def cut_data(obs_data, cwrf_data, lon_lat):
    """

    :param obs_data:
    :param cwrf_data:
    :param lon_lat:
    :return:
    """
    bound_arr = ~np.ma.getmaskarray(obs_data[0, 0].reshape(-1))
    obs_data = np.array(obs_data.reshape(obs_data.shape[0], obs_data.shape[1], -1).T)
    cwrf_data = np.array(cwrf_data.reshape(cwrf_data.shape[0], cwrf_data.shape[1], -1).T)
    idx = np.argwhere(bound_arr).reshape(-1)
    lon_lat = lon_lat[idx]

    cwrf_data = cwrf_data[idx].T
    obs_data = obs_data[idx].T
    obs_data = np.swapaxes(obs_data, 1, 2)
    cwrf_data = np.swapaxes(cwrf_data, 1, 2)
    return lon_lat.astype(np.float32), np.array(obs_data).astype(np.float32), \
           np.array(cwrf_data).astype(np.float32)


def load_temp_data(area="NCH", data_type="ORG", cwrf_case_name="01", case_time="00", year_num=29):
    father_data_dir = root_path

    cwrf_file_dir = father_data_dir
    cwrf_name = f"AT2M_c{cwrf_case_name}{case_time}_{area}_y1991-2020_{data_type}_monthly.nc"
    cwrf_file_path = os.path.join(cwrf_file_dir, cwrf_name)

    # obs_file_path = get_obs_file_path(area=area, date_type=data_type)

    obs_file_dir = father_data_dir
    obs_name = f"AT2M_obs_{area}_y1991-2019_{data_type}_monthly.nc"
    obs_file_path = os.path.join(obs_file_dir, obs_name)
    cwrf_nc_file = nc.Dataset(cwrf_file_path, 'r')
    # print(cwrf_nc_file)
    obs_nc_file = nc.Dataset(obs_file_path, 'r')

    if data_type == "AVG":
        obs_data = np.expand_dims(obs_nc_file.variables['VALUE'][:, :], axis=0)
        cwrf_data = np.expand_dims(cwrf_nc_file.variables['VALUE'][:, :], axis=0)
        cwrf_data = np.repeat(cwrf_data, year_num, axis=0)
        obs_data = np.repeat(obs_data, year_num, axis=0)
    else:
        obs_data = obs_nc_file.variables['VALUE'][:, :]
        cwrf_data = cwrf_nc_file.variables['VALUE'][:year_num, :]
    longitude = obs_nc_file.variables['longitude'][:].reshape(-1)
    latitude = obs_nc_file.variables['latitude'][:].reshape(-1)
    lon_lat = np.stack((longitude, latitude), axis=1)
    return cut_data(obs_data, cwrf_data, lon_lat)


def plot_one_me(year_list, cwrf_out, MBCp_model_out, MBCr_model_out, title="MSE"):
    """


    :return:
    """
    plt.title(title)
    plt.plot(year_list, cwrf_out, label="CWRF", c="black", linestyle='dashed')
    plt.plot(year_list, MBCp_model_out, label="MBCp", c="red")
    plt.plot(year_list, MBCr_model_out, label="MBCr", c="blue")
    plt.axhline(np.mean(cwrf_out), linestyle='--',  c="black")
    plt.axhline(np.mean(MBCp_model_out), c="red")
    plt.axhline(np.mean(MBCr_model_out), c="blue")
    plt.legend()
    plt.show()


def main(area="WHOLE", case_time="00", cwrf_case="02", year_num=29):
    """

    :return:
    """
    (lon_lat_array, obs_DPT, cwrf_DPT) = load_temp_data(area=area, data_type="DPT", case_time=case_time,
                                                        cwrf_case_name=cwrf_case, year_num=year_num)
    (lon_lat_array, obs_DPT_NCH, cwrf_DPT_NCH) = load_temp_data(area="NCH", data_type="DPT", case_time=case_time,
                                                                cwrf_case_name=cwrf_case, year_num=year_num)
    year_list = []
    model_out_mse = []
    MBCp_out_mse = []
    MBCr_out_mse = []

    model_out_corr = []
    MBCp_out_corr = []
    MBCr_out_corr = []

    for i in range(5, 29):
        year_list.append(int(i+1991))
        y_model_history = cwrf_DPT[i-5:i].reshape(-1, 5)
        y_model_obs_history = obs_DPT[i-5:i].reshape(-1, 5)

        y_model_predict = cwrf_DPT_NCH[i]
        y_model_obs_predict = obs_DPT_NCH[i]
        res = MBC.main(y_model_history, y_model_obs_history, y_model_predict, y_model_obs_predict)
        model_out_mse.append(res["model_out"]["MSE"])
        MBCp_out_mse.append(res["MBCp_out"]["MSE"])
        MBCr_out_mse.append(res["MBCr_out"]["MSE"])

        model_out_corr.append(res["model_out"]["corr"])
        MBCp_out_corr.append(res["MBCp_out"]["corr"])
        MBCr_out_corr.append(res["MBCr_out"]["corr"])

        print(f"{i+1991} :", res)
    plot_one_me(year_list, model_out_mse, MBCp_out_mse, MBCr_out_mse, title="MSE")
    plot_one_me(year_list, model_out_corr, MBCp_out_corr, MBCr_out_corr, title="ACC")


if __name__ == '__main__':
    main()
