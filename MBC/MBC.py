import random
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate
# from scipy import interpolate
from scipy import linalg
import multivariate_linear_bias_correction as mlbc
import univariate_quantile_mapping as uqm
import pandas as pd
import netCDF4 as nc


def containsNanArgsort(List, **kwargs):
    """
    获取含有nan值的列表的argsort
    :param List:
    :return:
    """
    # 先获取排好序的索引
    a = np.argsort(List, **kwargs)
    # 用字典的形式存储
    res = [(List[i], num) for num, i in enumerate(a)]
    res = dict([i for i in res if not np.isnan(i[0])])
    # 返回排序情况，若不在字典中则返回-1
    # print(res)
    res2 = []
    for i in List:
        if i in res.keys():
            res2.append(res[i])
        else:
            res2.append(-1)
    return res2


def MAE(x1, x2):
    return np.abs(x1 - x2).mean()


def MAE_cor(x_mh, x_oh, rowvar=False):
    """

    :return:
    """
    corr_mh = np.corrcoef(x_mh, rowvar=rowvar)
    corr_oh = np.corrcoef(x_oh, rowvar=rowvar)
    # print(corr_mh.shape)
    # print(corr_oh.shape)
    triu_idx = np.triu_indices(corr_mh.shape[0], k=1)
    corr_mh = corr_mh[triu_idx]
    corr_oh = corr_oh[triu_idx]
    return MAE(corr_mh, corr_oh) * 2


class MBCBase:
    def __init__(self, data_model, data_true):
        """

        :param data_model:
        :param data_true:
        """


def apply_uqm_models_by_dim(data_in, uqm_model_list):
    data_out = np.ones_like(data_in)
    for idx, n_uqm_model in enumerate(uqm_model_list):
        data_out[:, idx] = n_uqm_model(data_in[:, idx])
        # print(data_out[:, idx])
    return data_out


def mblc_progress(data_model, data_true, data_predict):
    """

    :param data_model:
    :param data_true:
    :param data_predict:
    :return:
    """
    mblc_model = mlbc.MultivariateRescaling(data_model, data_true)
    # data_model = mblc_model(data_model)
    # data_predict = mblc_model(data_predict)
    return mblc_model(data_model), mblc_model(data_predict)


class MBCp:
    def __init__(self, data_model, data_true, max_it=1000, uqm_type="QDM"):
        uqm_dict = {
            "QDM": uqm.QDM,
            "QM": uqm.QM
        }
        self.data_model = data_model
        self.data_true = data_true
        self.uqm_model = uqm_dict[uqm_type]
        self.max_it = max_it
        self.tol = 1e-13

    def get_uqm_models_by_dim(self, data_model, data_true):
        """

        :return:
        """
        data_dims = data_true.shape[1]
        uqm_model_list = []
        for i in range(data_dims):
            n_uqm_model = self.uqm_model(data_model[:, i], data_true[:, i])
            uqm_model_list.append(n_uqm_model)
        return uqm_model_list

    def uqm_progress(self, data_model, data_true, data_predict):
        """

        :param data_model:
        :param data_true:
        :param data_predict:
        :return:
        """
        uqm_models = self.get_uqm_models_by_dim(data_model, data_true)
        data_model = apply_uqm_models_by_dim(data_model, uqm_models)
        data_predict = apply_uqm_models_by_dim(data_predict, uqm_models)
        return data_model, data_predict

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def _mblc(self, data_model, data_true, data_predict):
        """

        :param data_model:
        :param data_true:
        :param data_predict:
        :return:
        """
        data_model, data_predict = mblc_progress(data_model, data_true, data_predict)
        data_model, data_predict = self.uqm_progress(data_model, data_true, data_predict)
        return data_model, data_predict

    def apply(self, data_predict):
        uqm_models = self.get_uqm_models_by_dim(self.data_model, self.data_true)
        data_model = apply_uqm_models_by_dim(self.data_model, uqm_models)
        data_predict = apply_uqm_models_by_dim(data_predict, uqm_models)
        mae_cor = MAE_cor(data_model, self.data_true)
        # print(data_model)
        # print(self.data_true)
        for epoch in range(self.max_it):
            data_model, data_predict = self._mblc(data_model, self.data_true, data_predict)
            mae_cor_ = MAE_cor(data_model, self.data_true)
            rate_ = mae_cor - mae_cor_
            mae_cor = mae_cor_
            if rate_ < self.tol:
                return data_model, data_predict
        return data_model, data_predict


class MBCr(MBCp):
    def __init__(self, *args, **kwargs):
        super(MBCr, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, data_predict):
        """

        :param data_predict:
        :return:
        """
        data_model_r = stats.rankdata(self.data_model, axis=0)
        data_true_r = stats.rankdata(self.data_true, axis=0)
        data_predict_r = stats.rankdata(data_predict, axis=0)
        mae_cor = MAE_cor(data_model_r, data_true_r)
        # print(data_model_r)
        # print(data_true_r)
        for it in range(self.max_it):
            data_model_r, data_predict_r = mblc_progress(data_model_r, data_true_r, data_predict_r)
            # print(data_model_r)
            data_model_r = stats.rankdata(data_model_r, axis=0)
            data_predict_r = stats.rankdata(data_predict_r, axis=0)
            mae_cor_ = MAE_cor(data_model_r, data_true_r)
            rate_ = mae_cor - mae_cor_
            mae_cor = mae_cor_
            # print(mae_cor_)
            if rate_ < self.tol:
                break
        data_model, data_predict = self.uqm_progress(self.data_model, self.data_true, data_predict)
        data_model = np.sort(data_model, axis=0)
        data_predict = np.sort(data_predict, axis=0)

        data_model_r = data_model_r.astype(int) - 1
        data_predict_r = data_predict_r.astype(int) - 1

        ndims = data_model.shape[1]
        for i in range(ndims):
            data_model[:, i] = data_model[:, i][data_model_r[:, i]]
            data_predict[:, i] = data_predict[:, i][data_predict_r[:, i]]
        return data_model, data_predict


def mse(y1, y2):
    # triu_idx = np.triu_indices(, k=1)
    co = []
    for i in range(y1.shape[1]):
        co.append(np.corrcoef(y1[i], y2[i])[0, 1])
    return {"MSE": round((np.abs(y1-y2)).mean(), 3),
            "corr": np.mean(co)
            }


def main():
    """

    :return:
    """
    # x = np.linspace(0, 10, 1000)
    x = np.random.normal(10, 5, size=(1000, 2))

    y_model = x**2 + 2*x + 1
    y_true = np.random.normal(0, 1, 100)
    y_true = x**3 + 2*x + 1
    # mr_model = MultivariateRescaling(model_data=y_model, true_data=y_true)
    #

    mr_model = MBCp(y_model, y_true)
    # mr_model = MBCr(y_model, y_true)

    #
    x = np.random.normal(10, 5, size=(1000, 2))
    y_model_predict = x ** 2 + 2 * x + 1
    y_true_predict = x**3 + 2*x + 1
    y_corr_predict = mr_model.apply(y_model_predict)
    #
    print(mse(y_true_predict, y_model_predict))
    print(mse(y_true_predict, y_corr_predict))


if __name__ == '__main__':
    main()


