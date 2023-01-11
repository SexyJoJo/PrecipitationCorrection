import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_CONST import *
from utils import OtherUtils, PaintUtils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import utils
from NN import NN
import matplotlib.colors


# 路径初始化
CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)
SHAPE = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR)).shape
MONTHS = utils.OtherUtils.get_predict_months(DATE, SHAPE[1])


class TestDataset(Dataset):
    def __init__(self, TEST_START_YEAR, TEST_END_YEAR):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA))
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, MONTHS))

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


def test():
    na_list = utils.ObsParser.get_na_index(OBS_DIR, AREA)
    for i in range(len(MONTHS)):
        corr_cases, test_cases, test_obses = [], [], []
        for TEST_YEAR in range(1991, 2020):
            # 读取模型
            model = NN()
            model.load_state_dict(torch.load(f"./models/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_1991-2019年模型(除{TEST_YEAR}年).pth"))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 取训练集中的最值用于反归一化
            train_case_data = torch.Tensor(
                utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR))
            train_obs_data = torch.Tensor(
                utils.ObsParser.get_many_2d_pravg(OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, MONTHS, JUMP_YEAR))
            all_train_data = torch.cat([train_case_data, train_obs_data], 0)
            tensor_min = torch.min(all_train_data)
            tensor_max = torch.max(all_train_data)

            # 加载测试集
            test_dataset = TestDataset(TEST_YEAR, TEST_YEAR)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

            with torch.no_grad():
                for data in test_dataloader:
                    data[0] = OtherUtils.min_max_normalization(data[0], tensor_min, tensor_max)
                    data[1] = OtherUtils.min_max_normalization(data[1], tensor_min, tensor_max)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # 反归一化
                    outputs = OtherUtils.min_max_denormalization(outputs, tensor_min, tensor_max)
                    inputs = OtherUtils.min_max_denormalization(inputs, tensor_min, tensor_max)
                    labels = OtherUtils.min_max_denormalization(labels, tensor_min, tensor_max)

                    # 去除obs中的缺失格点
                    # na_list = utils.ObsParser.get_na_index(OBS_DIR, AREA)
                    # for i, j in na_list:
                    #     input
                    print(f"{TEST_YEAR}年")
                    print("订正前mse", utils.OtherUtils.mse(inputs, labels))
                    print("订正后mse", utils.OtherUtils.mse(outputs, labels))

                    test_case = inputs[0][i].cpu().numpy()
                    corr_case = outputs[0][i].cpu().numpy()
                    test_obs = labels[0][i].cpu().numpy()
                    for a, b in na_list:
                        test_case[a][b] = np.nan
                        corr_case[a][b] = np.nan
                        test_obs[a][b] = np.nan
                    test_cases.append(test_case)
                    corr_cases.append(corr_case)
                    test_obses.append(test_obs)

                    plt.rcParams['font.family'] = ['SimHei']
                    fig = plt.figure()
                    fig.suptitle(f"{TIME}-{BASIN}-{AREA}-{TEST_YEAR}年{MONTHS[i]}月")
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)

                    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
                    ax1.set_title("订正前")
                    subfig = ax1.imshow(np.flip(inputs[0][i].numpy(), axis=0), norm=norm)

                    ax2.set_title("订正后")
                    ax2.imshow(np.flip(outputs[0][i].numpy(), axis=0), norm=norm)

                    ax3.set_title("obs")
                    ax3.imshow(np.flip(labels[0][i].numpy(), axis=0), norm=norm)

                    plt.colorbar(subfig, ax=[ax1, ax2, ax3], orientation="horizontal")

                    if not os.path.exists(rf"./results/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}"):
                        os.makedirs(rf"./results/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}")
                    plt.savefig(f"./results/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}/{TEST_YEAR}年{MONTHS[i]}月")
                    plt.close()

        # TCC相关
        corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)   # 订正后与真实值
        case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)   # 订正前与真实值
        tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
        if not os.path.exists(rf"./评价指标/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"):
            os.makedirs(rf"./评价指标/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}")
        tcc_img.savefig(rf"./评价指标/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[i]}月")
        print("tcc已保存", rf"./评价指标/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[i]}月")
        tcc_img.close()

        # ACC相关
        corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses)
        case_acc = OtherUtils.cal_ACC(test_cases, test_obses)
        acc_img = PaintUtils.paint_ACC(range(1991, 2020), case_acc, corr_acc)
        if not os.path.exists(rf"./评价指标/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"):
            os.makedirs(rf"./评价指标/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}")
        acc_img.savefig(f"./评价指标/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[i]}月")
        print("tcc已保存", rf"./评价指标/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_91-19年{MONTHS[i]}月")
        acc_img.close()


if __name__ == '__main__':
    plt.rcParams['axes.unicode_minus'] = False
    test()
