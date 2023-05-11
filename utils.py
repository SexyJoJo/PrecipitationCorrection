import calendar
import os
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.ma as ma
from dateutil.relativedelta import relativedelta
import torch


class CaseParser:
    @staticmethod
    def get_one_pravg(nc_path, month_index, remove_index=None):
        """
        提取单个case文件的某个月分的PRAVG值
        :param nc_path: nc文件路径
        :param month_index: 提取PRAVG所用的的月份索引
        :param remove_index: 需要删除的pravg格点索引
        :return: 一维PRAVG数组
        """
        case = netCDF4.Dataset(nc_path)
        pravg = case.variables["PRAVG"][month_index].flatten() * 24 * 3600  # 转换单位为mm/day
        if remove_index:
            for i in remove_index[::-1]:
                pravg = np.delete(pravg, i)
        return pravg

    @staticmethod
    def get_many_pravg(nc_dir, syear, eyear, area, month_index, remove_index=None):
        """
        提取指定时间、地区范围内的PRAVG组成一维数组
        :param nc_dir: 数据目录
        :param syear: 起始年份
        :param eyear: 结束年份
        :param area: 区域名称
        :param month_index: 提取PRAVG所用的的月份索引
        :param remove_index: 需要删除的pravg格点索引
        :return: 一维合并的PRAVG数组
        """
        stime = datetime(year=syear, month=1, day=1)
        etime = datetime(year=eyear + 1, month=1, day=1)

        pravgs = ma.masked_array([])
        for root, _, files in os.walk(nc_dir):
            for file in files:
                if file.startswith(area):
                    filetime = CaseParser.get_filetime(file)
                    if stime <= filetime <= etime:
                        pravg = CaseParser.get_one_pravg(os.path.join(root, file), month_index, remove_index)
                        pravgs = np.append(pravgs, pravg)
        return pravgs

    @staticmethod
    def get_one_2d_pravg(nc_path):
        """
        提取单个case文件的某个月分的PRAVG值(二维格点)
        :param nc_path: nc文件路径
        :return: 单个二维PRAVG数组
        """
        case = netCDF4.Dataset(nc_path)
        pravg = case.variables["PRAVG"][:] * 24 * 3600  # 转换单位为mm/day
        # if remove_index:
        #     for i in remove_index[::-1]:
        #         pravg = np.delete(pravg, i)
        return pravg

    @staticmethod
    def get_many_2d_pravg(nc_dir, syear, eyear, area, jump_years=None):
        """
        提取多个case文件的某个月分的PRAVG值(三维数组)
        :param jump_year: 跳过年份列表
        :param area: 区域简称
        :param eyear: 结束年份
        :param syear: 起始年份
        :param nc_dir: nc文件目录
        :return: 多个二维PRAVG数组（三维数组）
        """
        stime = datetime(year=syear, month=1, day=1)
        temp_jump_years = []
        if jump_years:
            for jump_year in jump_years:
                jump_year = datetime(year=jump_year, month=1, day=1)
                temp_jump_years.append(jump_year)
        etime = datetime(year=eyear + 1, month=1, day=1)

        pravgs = []
        for root, _, files in os.walk(nc_dir):
            for i, file in enumerate(files):
                if file.startswith(area):
                    filetime = CaseParser.get_filetime(file)
                    if jump_years:
                        if temp_jump_years[0] <= filetime <= temp_jump_years[0] + relativedelta(years=1) or \
                                temp_jump_years[1] <= filetime <= temp_jump_years[1] + relativedelta(years=1):
                            continue
                    if stime <= filetime <= etime:
                        pravg = CaseParser.get_one_2d_pravg(os.path.join(root, file))
                        pravgs.append(pravg)
        return np.array(pravgs)

    @staticmethod
    def get_filetime(filename):
        """
        从区域nc文件名中提取时间
        :param filename: 区域nc文件名
        :return: datetime
        """
        filetime = filename.split("_")[2][:-3]
        return datetime.strptime(filetime, "%Y%m%d%H")

    @staticmethod
    def get_filename(year, month_day, area, case_num):
        filename = f"{area}_PRAVG_{year + month_day}00c{case_num}_{year}_monthly.nc"
        return filename

    @staticmethod
    def data_enhance(all_arrays, step):
        """每个二维矩阵向8个方位移动step格"""

        def shift_2d_array(array):
            """每个二维矩阵向8个方位移动一格"""
            if len(array.shape) != 2:
                raise ValueError("Input array must be 2-dimensional")

            # Define 8 possible shifts
            shifts = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            shifted_arrays = []
            shifted_arrays.append(array)
            for dx, dy in shifts:
                shifted_array = np.roll(array, shift=(dx, dy), axis=(0, 1))

                # Fill empty edge positions with neighboring edge values
                if dx == 1:
                    shifted_array[0, :] = shifted_array[1, :]
                elif dx == -1:
                    shifted_array[-1, :] = shifted_array[-2, :]

                if dy == 1:
                    shifted_array[:, 0] = shifted_array[:, 1]
                elif dy == -1:
                    shifted_array[:, -1] = shifted_array[:, -2]

                shifted_arrays.append(shifted_array)

            return shifted_arrays

        def shift_1_and_2_steps(matrix):
            shifted_matrices = []

            for step in [1, 2]:
                for dx, dy in [(0, step), (0, -step), (step, 0), (-step, 0), (step, step), (-step, -step),
                               (step, -step), (-step, step)]:
                    shifted_matrix = np.roll(matrix, shift=(dx, dy), axis=(0, 1))

                    if dx > 0:
                        shifted_matrix[:dx, :] = shifted_matrix[dx:dx * 2, :]
                    elif dx < 0:
                        shifted_matrix[dx:, :] = shifted_matrix[2 * dx:dx, :]

                    if dy > 0:
                        shifted_matrix[:, :dy] = shifted_matrix[:, dy:dy * 2]
                    elif dy < 0:
                        shifted_matrix[:, dy:] = shifted_matrix[:, 2 * dy:dy]

                    shifted_matrices.append(shifted_matrix)

            return shifted_matrices

        new_all_arrays = []
        shape = all_arrays.shape
        all_arrays = all_arrays.reshape(-1, shape[2], shape[3])
        for array in all_arrays:
            if step == 1:
                shifted_arrays = shift_2d_array(array)
            elif step == 2:
                shifted_arrays = shift_1_and_2_steps(array)
            else:
                raise '不支持当前DATA_ENHANCE'
            new_all_arrays.append(shifted_arrays)

        new_all_arrays = np.array(new_all_arrays).reshape((-1, shape[1], shape[2], shape[3]))
        return new_all_arrays


class ObsParser:
    @staticmethod
    def get_one_pravg(nc_path, flatten=True):
        """
        提取时间范围
        :param nc_path:nc文件路径
        :param flatten: nc文件路径
        :return:
        """
        obs = netCDF4.Dataset(nc_path)
        pravg = obs.variables["prec"][:].flatten() / 30  # 转换单位为mm/day

        # 去除异常值
        remove_index = []
        for i in range(len(pravg)):
            if isinstance(pravg[i], np.ma.core.MaskedConstant):
                remove_index.append(i)
        for i in remove_index[::-1]:
            pravg = np.delete(pravg, i)

        return pravg, remove_index

    @staticmethod
    def get_many_pravg(nc_dir, syear, eyear, area, month, flatten=True):
        """
        提取指定时间、地区范围内的PRAVG组成一维数组
        :param nc_dir: 数据目录
        :param syear: 起始年份
        :param eyear: 结束年份
        :param area: 区域名称
        :param month: 提取月份
        :param flatten: 是否变为一维
        :return: 一维合并的PRAVG数组
        """
        stime = datetime(year=syear, month=1, day=1)
        etime = datetime(year=eyear + 1, month=1, day=1)
        pravgs = ma.masked_array([])
        for root, _, files in os.walk(nc_dir):
            for file in files:
                if file.startswith(area):
                    filetime = ObsParser.get_filetime(file)
                    if stime <= filetime <= etime and filetime.month == month:
                        pravg, remove_index = ObsParser.get_one_pravg(os.path.join(root, file), flatten)
                        pravgs = np.append(pravgs, pravg)
            return pravgs, remove_index

    @staticmethod
    def get_one_2d_pravg(nc_path, year, month):
        """
        提取时间范围
        :param year: 年份
        :param month: 月
        :param nc_path:nc文件路径
        :return:
        """
        obs = netCDF4.Dataset(nc_path)
        day_cnt = calendar.monthrange(int(year), int(month))[1]
        pravg = obs.variables["prec"][:] / day_cnt  # 转换单位为mm/day
        pravg = ObsParser.fill_na(pravg)
        return pravg

    @staticmethod
    def get_many_2d_pravg(nc_dir, syear, eyear, area, months, jump_years=None):
        """
        提取指定时间、地区范围内的PRAVG组成一维数组
        :param nc_dir: 数据目录
        :param syear: 起始年份
        :param eyear: 结束年份
        :param area: 区域名称
        :param months: 提取月份
        :param jump_years: 忽略的年份
        :return: 二维PRAVG数组
        """
        pravgs = []
        for year in range(syear, eyear + 1):
            if jump_years and year in jump_years:
                continue
            curr_year_pravg = []

            # 判断月份列表是否存在跨年的情况
            if months == sorted(months[:]):
                is_sorted = True
            else:
                is_sorted = False

            # 若case为12月
            if months[0] == 1:
                for month in months:
                    month = "0" + str(month) if len(str(month)) == 1 else str(month)
                    filename = area + "_obs_prec_rcm_" + str(year + 1) + month + ".nc"  # 年份加1
                    pravg = ObsParser.get_one_2d_pravg(os.path.join(nc_dir, filename), year, month)
                    curr_year_pravg.append(pravg)
            # 若case不为12月且月份列表顺序
            elif is_sorted and months[0] != 1:
                for month in months:
                    month = "0" + str(month) if len(str(month)) == 1 else str(month)
                    filename = area + "_obs_prec_rcm_" + str(year) + month + ".nc"  # 年份无需特殊处理
                    pravg = ObsParser.get_one_2d_pravg(os.path.join(nc_dir, filename), year, month)
                    curr_year_pravg.append(pravg)
            # 若月份列表非顺序
            elif not is_sorted:
                split_point = months.index(12)
                for i, month in enumerate(months):
                    month = "0" + str(month) if len(str(month)) == 1 else str(month)
                    if i <= split_point:
                        filename = area + "_obs_prec_rcm_" + str(year) + month + ".nc"  # 年份无需特殊处理
                    else:
                        filename = area + "_obs_prec_rcm_" + str(year + 1) + month + ".nc"  # 年份加1

                    pravg = ObsParser.get_one_2d_pravg(os.path.join(nc_dir, filename), year, month)
                    curr_year_pravg.append(pravg)

            pravgs.append(curr_year_pravg)
        return np.array(pravgs)
        # return ma.masked_array(pravgs)

    @staticmethod
    def fill_na(masked_array):
        """矩阵中的邻近格点值填充缺失值"""
        masked_array.fill_value = -9999
        masked_array = masked_array.filled()
        x = masked_array.copy()  # 替换后矩阵
        index_r = np.array(np.where(x == -9999)).T
        index_t = np.array(np.where(x != -9999)).T

        for ir in index_r:
            d = (ir - index_t) ** 2
            d = d[:, 0] + d[:, 1]
            p = index_t[np.argmin(d)]
            x[ir[0], ir[1]] = x[p[0], p[1]]
        return x

    @staticmethod
    def get_na_index(nc_dir, area):
        """
        获得obs缺失格点的坐标索引
        :return: 格点列表
        """
        for root, _, files in os.walk(nc_dir):
            for file in files:
                if file.startswith(area):
                    na_list, valid_list = [], []
                    nc_path = os.path.join(root, file)
                    obs = netCDF4.Dataset(nc_path)
                    pravg = obs.variables["prec"][:]
                    for i in range(len(pravg)):
                        for j in range(len(pravg[i])):
                            if isinstance(pravg[i][j], np.ma.core.MaskedConstant):
                                na_list.append((i, j))
                            else:
                                valid_list.append((i, j))
                    return na_list, valid_list

    @staticmethod
    def get_filetime(filename):
        """
        从区域nc文件名中提取时间
        :param filename: 区域nc文件名
        :return: datetime
        """
        filetime = filename.split("_")[4][:-3]
        return datetime.strptime(filetime, "%Y%m")

    @staticmethod
    def get_filename(year, month, area):
        filename = f"{area}_obs_prec_rcm_{year + month}.nc"
        return filename

    @staticmethod
    def data_enhance(all_arrays, step):
        def shift_2d_array(array):
            shifted_arrays = []
            for i in range(1 * 8 + 1):
                shifted_arrays.append(array)
            return shifted_arrays

        def shift_1_and_2_steps(array):
            shifted_arrays = []
            for i in range(2 * 8 + 1):
                shifted_arrays.append(array)
            return shifted_arrays

        new_all_arrays = []
        shape = all_arrays.shape
        all_arrays = all_arrays.reshape(-1, shape[2], shape[3])
        for array in all_arrays:
            if step == 1:
                shifted_arrays = shift_2d_array(array)
            elif step == 2:
                shifted_arrays = shift_1_and_2_steps(array)
            else:
                raise '不支持当前DATA_ENHANCE'
            new_all_arrays.append(shifted_arrays)

        new_all_arrays = np.array(new_all_arrays).reshape((-1, shape[1], shape[2], shape[3]))
        return new_all_arrays


class OtherUtils:
    @staticmethod
    def get_predict_months(case_date, month_cnt):
        """
        获取需要预测月份的列表
        :param case_date: case数据起始日期
        :param month_cnt: 待预测月份个数
        :return: 待预测月份列表
        """
        start_month = int(case_date[:2])
        months = []
        for cnt in range(month_cnt):
            months.append((start_month + cnt + 1) % 13)

        if 0 in months:
            idx = months.index(0)
            months.append(months[-1] + 1)
            months.pop(idx)
        return months

    @staticmethod
    def trans_time(nc_time):
        start = datetime(1900, 1, 1)
        delta = timedelta(hours=int(nc_time))
        return start + delta

    @staticmethod
    def cal_TCC(cases, obses):
        """
        计算时间相关系数TCC
        :param cases: 预测值， 维度：[年份， 预测值]
        :param obses: 对应预测值的观测值， 维度：[年份， 观测值]
        :return: TCC(二维 一个格点对应一个值)
        """
        avg_case = np.mean(cases, axis=0)
        avg_obs = np.mean(obses, axis=0)

        # TCC公式分子
        molecular = np.zeros(cases[0].shape)
        # 分母左半边， 分母右半边
        denominator_left, denominator_right = np.zeros(cases[0].shape), np.zeros(cases[0].shape)

        for i in range(len(cases)):
            molecular += (cases[i] - avg_case) * (obses[i] - avg_obs)
            denominator_left += np.square(cases[i] - avg_case)
            denominator_right += np.square(obses[i] - avg_obs)

        TCC = molecular / np.sqrt(denominator_left * denominator_right)

        # shape = cases[0].shape
        # x, y = [], []
        # for year in range(len(cases)):
        #     case = cases[year].flatten()
        #     obs = obses[year].flatten()
        #     x.append(case)
        #     y.append(obs)
        # x, y = np.array(x), np.array(y)
        #
        # TCCs = []
        # for i in range(x.shape[1]):     # shape[1]: 格点数
        #     case_avg = np.mean(x[:, i])
        #     obs_avg = np.mean(y[:, i])
        #     case_std = np.std(x[:, i])
        #     obs_std = np.std(y[:, i])
        #
        #     sum = 0
        #     for j in range(x.shape[0]):     # shape[0]: 年数
        #         a = (x[j, i] - case_avg) * (y[j, i] - obs_avg)
        #         if np.isnan(a):
        #             continue
        #         sum += a
        #
        #     tcc = (1 / x.shape[1] * sum) / (case_std * obs_std)
        #     TCCs.append(tcc)
        # TCCs = np.array(TCCs)
        # TCCs = TCCs.reshape(shape)
        return TCC

    @staticmethod
    def cal_ACC(cases, obses, use_anomaly=True):
        """
        计算距平相关系数ACC
        :param cases: 所有年份的订正结果
        :param obses: 所有年份订正结果对应的观测值
        :param use_anomaly: 是否使用距平百分率，默认使用
        :return: ACC列表（一维 每个年份对应一个值）
        """
        # corr_cases = np.array(corr_cases)
        # test_obses = np.array(test_obses)
        #
        # if use_anomaly:
        #     # 各个格点转为距平百分率
        #     deltaRfs, deltaRos = [], []
        #     for i in range(len(corr_cases)):
        #         deltaRf = OtherUtils.cal_anomaly_percentage(corr_cases[i], corr_cases)  # 预测的距平百分率
        #         deltaRfs.append(deltaRf)
        #         deltaRo = OtherUtils.cal_anomaly_percentage(test_obses[i], test_obses)  # obs的距平百分率
        #         deltaRos.append(deltaRo)
        #
        #     deltaRfs = np.array(deltaRfs)
        #     avg_deltaRf = np.mean(deltaRfs, axis=0)
        #     deltaRos = np.array(deltaRos)
        #     avg_deltaRo = np.mean(deltaRos, axis=0)
        # else:
        #     deltaRfs = corr_cases
        #     deltaRos = test_obses
        # # i对应各预测年份
        # ACCs = []
        # for i in range(len(corr_cases)):
        #     # ACC公式分子
        #     avg_deltaRf = np.nanmean(deltaRfs[i])
        #     avg_deltaRo = np.nanmean(deltaRos[i])
        #
        #     molecular = np.nansum((deltaRfs[i] - avg_deltaRf) * (deltaRos[i] - avg_deltaRo))
        #     denominator_left = np.nansum(np.square(deltaRfs[i] - avg_deltaRf))
        #     denominator_right = np.nansum(np.square(deltaRos[i] - avg_deltaRo))
        #     ACC = molecular / np.sqrt(denominator_left * denominator_right)
        #     ACCs.append(ACC)

        ACCs = []
        for year in range(len(cases)):
            case = cases[year].flatten()
            obs = obses[year].flatten()
            nan_mask1 = np.isnan(case)
            nan_mask2 = np.isnan(obs)
            case = case[~nan_mask1]
            obs = obs[~nan_mask2]
            point_num = len(obs)

            case_avg = np.mean(case)
            obs_avg = np.mean(obs)
            case_std = np.std(case)
            obs_std = np.std(obs)

            sum = 0
            for i in range(point_num):
                a = (case[i] - case_avg) * (obs[i] - obs_avg)
                sum += a

            acc = (1 / point_num * sum) / (case_std * obs_std)
            ACCs.append(acc)
        return ACCs

    @staticmethod
    def cal_anomaly_percentage(pre, all_pre):
        """
        计算距平百分率, （实测值-同期历史均值）/同期历史均值）
        :param pre: 当年预测值
        :param all_pre: 全部年的预测值
        :return: 距平百分率
        """
        all_pre = np.array(all_pre)
        avg_pre = np.mean(all_pre, axis=0)
        result = (pre - avg_pre) / avg_pre
        return result

    @staticmethod
    def d_2_2d(d, length):
        two_d = d.reshape(int(len(d) / length), length)
        return two_d

    @staticmethod
    def map2grid(data, valid_girds, year_cnt):
        samples = []
        for year in range(year_cnt):
            for i, j in valid_girds:
                samples.append(data[year, :, i, j].tolist())
        return torch.Tensor(samples)

    @staticmethod
    def map2grid33(data, valid_girds, year_cnt):
        def get_neighbors(case_data, row, col):
            neighbors = []
            for r in range(row - 1, row + 2):
                for c in range(col - 1, col + 2):
                    if 0 <= r < len(case_data) and 0 <= c < len(case_data[0]):
                        value = case_data[r][c]
                        if np.isnan(value):
                            return []
                        neighbors.append(value)
            if len(neighbors) == 9:  # 8邻居加上自身
                return neighbors
            else:
                return []

        samples = []
        new_valid_girds = []
        for year in range(year_cnt):
            for i, j in valid_girds:
                all_month_neighbors = []
                for month in range(data.shape[1]):
                    neighbors = get_neighbors(data[year][month], i, j)
                    all_month_neighbors += neighbors
                if len(all_month_neighbors) != 0:
                    samples.append(all_month_neighbors)
                    if (i, j) not in new_valid_girds:
                        new_valid_girds.append((i, j))
        return torch.Tensor(samples), new_valid_girds

    @staticmethod
    def cal_mse(y1, y2):
        return np.nanmean(np.square(y1 - y2))

    @staticmethod
    def cal_mean_std(data):
        """计算各月均值与标准差"""
        means = []
        stds = []
        for m in range(data.shape[1]):
            means.append(np.nanmean(data[:, m, :, :]))
            stds.append(np.nanstd(data[:, m, :, :]))
        return means, stds

    @staticmethod
    def zscore_normalization(tensor, means, stds):
        """Z-Score归一化（各月）"""
        for m in range(tensor.shape[1]):
            tensor[:, m, :, :] = (tensor[:, m, :, :] - means[m]) / stds[m]
        return tensor

    @staticmethod
    def zscore_denormalization(tensor, means, stds, data_format):
        """Z-Score归一化（各月）"""
        if data_format == 'map':
            for m in range(tensor.shape[1]):
                tensor[:, m, :, :] = tensor[:, m, :, :] * stds[m] + means[m]
        elif data_format.startswith('grid'):
            for m in range(tensor.shape[0]):
                tensor[m] = tensor[m] * stds[m] + means[m]
        return tensor

    @staticmethod
    def min_max_normalization(tensor, tensor_min, tensor_max):
        """MinMax归一化"""
        return (tensor - tensor_min) / (tensor_max - tensor_min)

    @staticmethod
    def min_max_denormalization(tensor, tensor_min, tensor_max):
        """MinMax反归一化"""
        return (tensor_max - tensor_min) * tensor + tensor_min


class PaintUtils:
    @staticmethod
    def paint_ACC(xrange, case_acc, corr_case_acc):
        """
        ACC绘图
        :param xrange: x轴年范围
        :param corr_case_acc:
        :param case_acc:
        :return: 图片对象
        """
        plt.ylim(-1, 1)
        case_line, = plt.plot(list(xrange), case_acc, linestyle='-', color='k', marker='^', markersize=7)
        case_avg = np.mean(case_acc)
        plt.axhline(y=case_avg, color='k', linestyle=':')

        corr_case_line, = plt.plot(list(xrange), corr_case_acc, linestyle='-', color='r', marker='.', markersize=7)
        corr_case_avg = np.mean(corr_case_acc)
        plt.axhline(y=corr_case_avg, color='r', linestyle=':')

        plt.legend(handles=[case_line, corr_case_line], labels=["CASE回报", "订正后CASE"], loc="lower right")
        return plt

    @staticmethod
    def paint_TCC(case_tcc, corr_tcc):
        plt.rcParams['font.family'] = ['SimHei']
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)  # 设置colorbar显示的最大最小值

        fig = plt.figure(figsize=(7, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title("订正前TCC")
        sub_fig1 = ax1.imshow(np.flip(case_tcc, axis=0), cmap='bwr', norm=norm)
        ax2.set_title("订正后TCC")
        ax2.imshow(np.flip(corr_tcc, axis=0), cmap='bwr', norm=norm)

        fig.colorbar(sub_fig1, ax=[ax1, ax2], orientation="horizontal")
        return plt

    @staticmethod
    def paint_hist(before, after, obs, bins):
        """
        绘制数据分布直方图
        :param before: 订正前
        :param after: 订正后
        :param obs: 原始值
        :param bins: 图柱间隔

        :return:
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        plt.title("订正前")
        plt.hist(before, bins, alpha=0.5, label="订正前")
        plt.show()
        plt.close()

        plt.title("订正后")
        plt.hist(after, bins, alpha=0.5, label="订正后")
        plt.show()
        plt.close()

        plt.title("原始值")
        plt.hist(obs, bins, alpha=0.5, label="观测值")
        plt.show()
        plt.close()
        # plt.legend(loc='lower right')
        return plt

    @staticmethod
    def paint_result(title, img_path, input, output, label):
        plt.rcParams['font.family'] = ['SimHei']
        fig = plt.figure()
        fig.suptitle(title)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
        ax1.set_title("订正前")
        subfig = ax1.imshow(np.flip(input.numpy(), axis=0), norm=norm)
        ax2.set_title("订正后")
        ax2.imshow(np.flip(output.numpy(), axis=0), norm=norm)
        ax3.set_title("obs")
        ax3.imshow(np.flip(label.numpy(), axis=0), norm=norm)
        plt.colorbar(subfig, ax=[ax1, ax2, ax3], orientation="horizontal")

        os.makedirs(img_path, exist_ok=True)
        plt.savefig(f"{img_path}/{title}")
        plt.close()


if __name__ == '__main__':
    for root, dirs, files in os.walk(r"C:\Users\JOJO\Desktop\三亚EC数据_探空格式"):
        for file in files:
            new_file = file.replace('11111', '12345')
            if file.startswith('11111'):
                os.rename(os.path.join(root, file), os.path.join(root, new_file))
