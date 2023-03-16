import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_CONST import *
from utils import OtherUtils, PaintUtils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import utils
from train import TestDataset, TrainDataset
import matplotlib.colors
from models import *

# 路径初始化
CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)
SHAPE = np.ones((1, 5, 3, 3)).shape
MONTHS = utils.OtherUtils.get_predict_months(DATE, 1)


def test():
    # na_list = utils.ObsParser.get_na_index(OBS_DIR, AREA)
    corr_cases, test_cases, test_obses = [], [], []
    for TEST_YEAR in range(1991, 2020):
        # 输出数组初始化
        inputs_map = np.zeros([1, 5, 43, 39]) + np.nan
        outputs_map = np.zeros([1, 1, 43, 39]) + np.nan
        labels_map = np.zeros([1, 1, 43, 39]) + np.nan

        # 读取模型
        # model = LSTM(input_size=9, hidden_size=64, num_layers=1, output_size=1)
        model = ANN()
        model.load_state_dict(
            torch.load(MODEL_PATH + f"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_1991-2019年模型(除{TEST_YEAR}年).pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载训练集
        train_dataset = TrainDataset(TEST_YEAR)
        # 加载测试集
        test_dataset = TestDataset(TEST_YEAR, TEST_YEAR, train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        for m in range(len(MONTHS)):
            with torch.no_grad():
                # 处理每个格点
                for cnt, data in enumerate(test_dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # 反归一化
                    outputs = OtherUtils.min_max_denormalization(outputs,
                                                                 train_dataset.data_min, train_dataset.data_max)
                    inputs = OtherUtils.min_max_denormalization(inputs,
                                                                train_dataset.data_min, train_dataset.data_max)
                    labels = OtherUtils.min_max_denormalization(labels,
                                                                train_dataset.data_min, train_dataset.data_max)

                    # outputs = OtherUtils.zscore_denormalization(outputs,
                    #                                             train_dataset.case_mean, train_dataset.case_std)
                    # inputs = OtherUtils.zscore_denormalization(inputs,
                    #                                            train_dataset.case_mean, train_dataset.case_std)
                    # labels = OtherUtils.zscore_denormalization(labels,
                    #                                            train_dataset.case_mean, train_dataset.case_std)

                    a, b = train_dataset.valid_indexes[cnt]
                    inputs_map[0][m][a][b] = inputs[0][m][1][1]
                    outputs_map[0][m][a][b] = outputs[0][0][0]
                    labels_map[0][m][a][b] = labels[0][0][0]

            print(f"{TEST_YEAR}年{MONTHS[m]}月")
            print("订正前mse", utils.OtherUtils.mse(inputs_map[0][m], labels_map[0][m]))
            print("订正后mse", utils.OtherUtils.mse(outputs_map[0][m], labels_map[0][m]))

            test_case = inputs_map[0][m]
            corr_case = outputs_map[0][m]
            test_obs = labels_map[0][m]

            if USE_ANOMALY:
                test_cases.append(test_case + train_dataset.case_avg[m])
                corr_cases.append(corr_case + train_dataset.case_avg[m])
                test_obses.append(test_obs + train_dataset.obs_avg[m])
                norm = matplotlib.colors.Normalize(vmin=-5, vmax=5)
            else:
                test_cases.append(test_case)
                corr_cases.append(corr_case)
                test_obses.append(test_obs)
                norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

            plt.rcParams['font.family'] = ['SimHei']
            fig = plt.figure()
            fig.suptitle(f"{TIME}-{BASIN}-{AREA}-{TEST_YEAR}年{MONTHS[m]}月")
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.set_title("订正前")
            subfig = ax1.imshow(np.flip(inputs_map[0][m], axis=0), norm=norm)

            ax2.set_title("订正后")
            ax2.imshow(np.flip(outputs_map[0][m], axis=0), norm=norm)

            ax3.set_title("obs")
            ax3.imshow(np.flip(labels_map[0][m], axis=0), norm=norm)

            plt.colorbar(subfig, ax=[ax1, ax2, ax3], orientation="horizontal")

            if not os.path.exists(RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}"):
                os.makedirs(RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}")
            plt.savefig(RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}/{TEST_YEAR}年{MONTHS[m]}月")
            plt.close()

    # TCC相关
    corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)  # 订正后与真实值
    case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)  # 订正前与真实值
    tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
    if not os.path.exists(EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"):
        os.makedirs(EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}")
    tcc_img.savefig(EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[m]}月")
    print("tcc已保存", rf"{EVALUATE_PATH}/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[m]}月")
    tcc_img.close()

    # ACC相关
    corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses)
    case_acc = OtherUtils.cal_ACC(test_cases, test_obses)
    acc_img = PaintUtils.paint_ACC(range(1991, 2020), case_acc, corr_acc)
    if not os.path.exists(EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"):
        os.makedirs(EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}")
    acc_img.savefig(EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[m]}月")
    print("tcc已保存", rf"{EVALUATE_PATH}/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[m]}月")
    acc_img.close()


if __name__ == '__main__':
    plt.rcParams['axes.unicode_minus'] = False
    test()
