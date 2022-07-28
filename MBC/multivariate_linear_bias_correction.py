import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate
# from scipy import interpolate
from scipy import linalg


def get_L(data_in):
    return linalg.cholesky(np.cov(data_in.T))


def mse(y1, y2):
    # triu_idx = np.triu_indices(, k=1)
    co = []
    for i in range(y1.shape[1]):
        co.append(np.corrcoef(y1[i], y2[i])[0, 1])
    return {"MSE": round((np.abs(y1-y2)).mean(), 3),
            "corr": np.mean(co)
            }


def cor(y1, y2):

    return np.corrcoef(y1, y2)


class UnivariateRescaling:
    def __init__(self, model_data, true_data):
        self.model_data_mean = model_data.mean()
        model_data_std = model_data.std(ddof=1)
        self.true_data_mean = true_data.mean()
        true_data_std = true_data.std(ddof=1)
        self.K = true_data_std/model_data_std

    def __call__(self,  *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, data_in):
        data_mean = data_in.mean()
        data_s = data_in - data_mean
        data_s_hat = data_s * self.K
        y_corr = data_s_hat + self.true_data_mean + data_mean - self.model_data_mean
        return y_corr


class MultivariateRescaling:
    def __init__(self, model_data, true_data):

        self.model_data_mean = model_data.mean(axis=0, keepdims=True)
        self.true_data_mean = true_data.mean(axis=0, keepdims=True)

        self.model_data_s = (model_data - self.model_data_mean)
        self.true_data_s = (true_data - self.true_data_mean)

        self.model_L = get_L(self.model_data_s)
        model_L_I = np.linalg.inv(self.model_L)
        true_L = get_L(self.true_data_s)
        self.K = model_L_I @ true_L

    def __call__(self,  *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, data_inx):
        data_mean = data_inx.mean(axis=0, keepdims=True)
        data_s = (data_inx - data_mean)
        data_s_hat = data_s @ self.K
        data_corr = data_s_hat + self.true_data_mean + data_mean - self.model_data_mean
        return data_corr


def main():
    """

    :return:
    """
    # x = np.linspace(0, 10, 1000)
    x = np.random.normal(10, 5, size=(1000, 20))  # 输入历史x
    y_model = x**2 + 2*x + 1
    # y_true = np.random.normal(0, 1, 100)
    y_true = x**3 + 2*x + 1  # 输入历史y  obs
    # mr_model = MultivariateRescaling(model_data=y_model, true_data=y_true)
    mr_model = UnivariateRescaling(model_data=y_model, true_data=y_true)

    x = np.random.normal(10, 5, size=(1000, 20))
    y_model_predict = x ** 2 + 2 * x + 1
    y_true_predict = x**3 + 2*x + 1
    y_corr_predict = mr_model.predict(y_model_predict)  # 输入当前x, 得到订正后的y
    print(y_corr_predict.shape)
    print(mse(y_true_predict, y_model_predict))
    print(mse(y_true_predict, y_corr_predict))


if __name__ == '__main__':
    main()
    # x = np.random.random(size=(1000, 3)).T
    # x = np.ones(shape=(1000, 4)).T
    # co = np.cov(x)
    # cor = np.corrcoef(x)
    # print(co.shape)
    # print(co)
    # print(cor)
    # arr1 = np.array([[2. + 0.j, -0. - 3.j], [0. + 3.j, 5. + 0.j]])
    # print("Original array is :\n", arr1)
    # print("Original array is :\n", arr1.shape)
    # L = np.linalg.cholesky(co).T
    # print(L)
    # L = linalg.cholesky(co)
    # L_cor = linalg.cholesky(cor)
    # print(L_cor)
    # print(L)
    # print(L.shape)

