import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import numpy.ma as ma


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


class OtherUtils:
    @staticmethod
    def trans_time(nc_time):
        start = datetime(1900, 1, 1)
        delta = timedelta(hours=int(nc_time))
        return start + delta

    @staticmethod
    def cal_TCC(predicts, obses):
        """
        计算时间相关系数TCC
        :param predicts: 预测值， 维度：[年份， 预测值]
        :param obses: 对应预测值的观测值， 维度：[年份， 观测值]
        :return: TCC(一个格点对应一个值)
        """
        avg_pre = np.mean(predicts, axis=0)
        avg_obs = np.mean(obses, axis=0)

        # TCC公式分子
        molecular = np.zeros([len(predicts[0])])
        # 分母左半边， 分母右半边
        denominator_left, denominator_right = np.zeros([len(predicts[0])]), np.zeros([len(predicts[0])])

        for i in range(len(predicts)):
            molecular += (predicts[i] - avg_pre) * (obses[i] - avg_obs)
            denominator_left += np.square(predicts[i] - avg_pre)
            denominator_right += np.square(obses[i] - avg_obs)

        TCC = molecular / np.sqrt(denominator_left * denominator_right)
        return TCC

    @staticmethod
    def cal_ACC(corr_cases, test_obses):
        """
        计算距平相关系数ACC
        :param corr_cases: 所有年份的订正结果
        :param test_obses: 所有年份订正结果对应的观测值
        :return: ACC列表（每个年份对应一个值）
        """
        deltaRfs, deltaRos = [], []
        for i in range(len(corr_cases)):
            deltaRf = OtherUtils.cal_anomaly_percentage(corr_cases[i], corr_cases)
            deltaRfs.append(deltaRf)
            deltaRo = OtherUtils.cal_anomaly_percentage(test_obses[i], test_obses)
            deltaRos.append(deltaRo)

        deltaRfs = ma.masked_array(deltaRfs)
        avg_deltaRf = np.mean(deltaRfs, axis=0)
        deltaRos = ma.masked_array(deltaRos)
        avg_deltaRo = np.mean(deltaRos, axis=0)

        # i对应各预测年份
        ACCs = []
        for i in range(len(corr_cases)):
            # ACC公式分子
            molecular = np.sum((deltaRfs[i] - avg_deltaRf) * (deltaRos[i] - avg_deltaRo))
            denominator_left = np.sum(np.square(deltaRfs[i] - avg_deltaRf))
            denominator_right = np.sum(np.square(deltaRos[i] - avg_deltaRo))
            ACC = molecular / np.sqrt(denominator_left * denominator_right)
            ACCs.append(ACC)
        return ACCs

    @staticmethod
    def cal_anomaly_percentage(pre, all_pre):
        """
        计算距平百分率, （实测值-同期历史均值）/同期历史均值）
        :param pre: 当年预测值
        :param all_pre: 全部年的预测值
        :return: 距平百分率
        """
        all_pre = ma.masked_array(all_pre)
        avg_pre = np.mean(all_pre, axis=0)
        return (pre - avg_pre) / avg_pre

    @staticmethod
    def d_2_2d(d, length):
        two_d = d.reshape(int(len(d) / length), length)
        return two_d


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
        corr_case_line, = plt.plot(list(xrange), corr_case_acc, linestyle='-', color='r', marker='.', markersize=7)
        plt.legend(handles=[case_line, corr_case_line], labels=["CASE回报", "单变量订正"], loc="lower right")
        return plt

    @staticmethod
    def paint_TCC(tcc, label=None):
        avg = np.mean(tcc)
        plt.axhline(y=avg, color='b', linestyle=':')
        plt.plot(tcc, label=label)
        plt.legend()
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
