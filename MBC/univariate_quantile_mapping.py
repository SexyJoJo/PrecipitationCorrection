import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from scipy import stats, interpolate
from utils import *
import netCDF4


def get_cdf(data_in):
    """

    :return:
    """
    # res_freq = stats.relfreq(data_in, numbins=data_in.shape[0])
    # cdf_value = stats.norm.cdf(data_in)
    # mean = data_in.mean()
    # print(mean)
    # std = data_in.std(ddof=1)
    # cdf = stats.norm.cdf(data_in, loc=mean, scale=std)

    # print(std)
    data_arg = np.argsort(data_in)
    # cdf_value = np.cumsum(res_freq[0])
    sorted_random_data = data_in[data_arg]
    cdf_value = 1. * np.arange(len(sorted_random_data)) / float(len(sorted_random_data) - 1)
    cdf = np.zeros_like(cdf_value)
    cdf[data_arg] = cdf_value
    return cdf


def get_cdf_func(p_cdf, data_in):
    """

    :return:
    """
    # p_cdf = get_cdf(data_in)
    # print(p_cdf)
    inter = interpolate.interp1d(data_in, p_cdf, kind="cubic")
    return inter


def get_icdf_func(p_cdf, data_in):
    """

    :return:
    """
    # p_cdf = get_cdf(data_in)
    inter = interpolate.interp1d(p_cdf, data_in, kind="cubic", bounds_error=False, fill_value=(0., 1.))
    return inter


class QM:
    def __init__(self, model_data, true_data):
        true_data_cdf = get_cdf(true_data)
        self.true_data_icdf_func = get_icdf_func(true_data_cdf, true_data)

        model_data_cdf = get_cdf(model_data)
        self.model_data_cdf_func = get_cdf_func(model_data_cdf, model_data)
        # self.model_data_icdf_func = get_icdf_func(model_data_cdf, model_data)
        self.model_data_range = [model_data.min(), model_data.max()]

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, data_in):
        x = data_in.copy()
        x[x > self.model_data_range[1]] = self.model_data_range[1]
        x[x < self.model_data_range[0]] = self.model_data_range[0]
        x_cdf = self.model_data_cdf_func(x)
        x_cdf[x_cdf > 1.] = 1.
        x_cdf[x_cdf < 0.] = 0.
        x_corr = self.true_data_icdf_func(x_cdf)
        return x_corr


class QDM:
    def __init__(self, model_data, true_data):
        true_data_cdf = get_cdf(true_data)
        # print(true_data_cdf)
        # print(true_data)
        self.true_data_icdf_func = get_icdf_func(true_data_cdf, true_data)
        # stats.norm.isf()

        model_data_cdf = get_cdf(model_data)
        self.model_data_icdf_func = get_icdf_func(model_data_cdf, model_data)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, data_in):
        data_in_cdf = get_cdf(data_in)
        # data_in_cdf[data_in_cdf > 1.] = 1.
        # data_in_cdf[data_in_cdf < 0.] = 0.
        # print(data_in_cdf)
        x_corr = self.true_data_icdf_func(data_in_cdf) + data_in - self.model_data_icdf_func(data_in_cdf)
        # print(x_corr)
        return x_corr


def mse(y1, y2):
    return (np.square(y1 - y2)).mean()


def single_train():
    """

    :return:
    """
    obs, remove = ObsParser.get_many_pravg(r"../divide area/divided obs", 2013, 2018, "JSJ", 4)
    case = CaseParser.get_many_pravg(r"../divide area/divided case", 2013, 2018, "JSJ", 0, remove)

    test_obs_filename = ObsParser.get_filename("2019", "04", "JSJ")
    test_case_filename = CaseParser.get_filename("2019", "0302", "JSJ", "01")
    test_obs, _ = ObsParser.get_one_pravg(os.path.join(r"../divide area/divided obs/", test_obs_filename))
    test_case = CaseParser.get_one_pravg(os.path.join(r"../divide area/divided case", test_case_filename), 0, remove)

    uqdm = QDM(case, obs)
    model = uqdm.predict(test_case)  # 得到订正后的y

    print("mse(obs, case):", mse(obs, case))
    print("mse(test_obs, model):", mse(test_obs, model))
    print("np.corrcoef(obs, case):", np.corrcoef(obs, case))
    print("np.corrcoef(test_obs, model):", np.corrcoef(test_obs, model))


def loop_train():
    corr_cases, test_obses = [], []
    for syear in range(1991, 2015):
        # 训练集
        #   真实值
        obs, remove = ObsParser.get_many_pravg(r"../divide area/divided obs", syear, syear+4, "JSJ", 4)
        case = CaseParser.get_many_pravg(r"../divide area/divided case", syear, syear+4, "JSJ", 0, remove)

        # 测试集
        test_obs_filename = ObsParser.get_filename(str(syear+5), "04", "JSJ")
        test_case_filename = CaseParser.get_filename(str(syear+5), "0302", "JSJ", "01")
        test_obs, _ = ObsParser.get_one_pravg(os.path.join(r"../divide area/divided obs", test_obs_filename))
        test_case = CaseParser.get_one_pravg(os.path.join(r"../divide area/divided case", test_case_filename), 0, remove)

        # 训练模型
        uqdm = QDM(case, obs)

        # 订正
        corr_case = uqdm.predict(test_case)  # 得到订正后的case

        # 绘制数据分布折线图
        # plt.plot(test_case, color='blue')
        # plt.plot(corr_case, color='orange')
        # plt.plot(test_obs, color="black")
        # plt.show()
        # plt.close()

        # 绘制数据分布直方图
        hist_img = PaintUtils.paint_hist(test_case, corr_case, test_obs, bins=np.linspace(-10, 40, 25))
        # hist_img.show()

        corr_cases.append(corr_case)
        test_obses.append(test_obs)

        print("---------------------")
        print(f"year range:{syear}-{syear+4}")
        print("mse(obs, case):", mse(obs, case))
        print("mse(test_obs, model):", mse(test_obs, corr_case))
        print("np.corrcoef(obs, case):", np.corrcoef(obs, case))
        print("np.corrcoef(test_obs, model):", np.corrcoef(test_obs, corr_case))

    all_obs, remove = ObsParser.get_many_pravg(r"../divide area/divided obs", 1991, 2018, "JSJ", 4)
    all_cases = CaseParser.get_many_pravg(r"../divide area/divided case", 1991, 2018, "JSJ", 0, remove)
    tcc = OtherUtils.cal_TCC(corr_cases, all_cases, test_obses, all_obs)

    # ACC相关
    part_obses, remove = ObsParser.get_many_pravg(r"../divide area/divided obs", 1996, 2019, "JSJ", 4)
    part_cases = CaseParser.get_many_pravg(r"../divide area/divided case", 1996, 2019, "JSJ", 0, remove)
    part_obses = OtherUtils.d_2_2d(part_obses, len(corr_cases[0]))  # 1996 - 2019
    part_cases = OtherUtils.d_2_2d(part_cases, len(corr_cases[0]))  # 1996 - 2019

    case_acc = OtherUtils.cal_ACC(part_cases, part_obses)
    corr_case_acc = OtherUtils.cal_ACC(corr_cases, test_obses)
    # acc_img = PaintUtils.paint_ACC(range(1996, 2020), case_acc, corr_case_acc)
    # acc_img.show()

    print("---------------------")
    print("TCC:", tcc)
    print("ACC:", corr_case_acc)
    print("CASE ACC:", case_acc)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # single_train()
    loop_train()