import matplotlib.pyplot as plt
import numpy as np
import torch
from NN_CONST import *
from utils import OtherUtils, PaintUtils
from torch.utils.data import Dataset, DataLoader
from model import *
from dataset import *
import matplotlib.colors

SHAPE = torch.Tensor(
    utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_START_YEAR+1, AREA)).shape
MONTHS = utils.OtherUtils.get_predict_months(DATE, SHAPE[1])


def test():
    na_list = utils.ObsParser.get_na_index(OBS_DIR, AREA)
    syear, eyear = 1991, 2001
    for i in range(len(MONTHS)):
        corr_cases, test_cases, test_obses = [], [], []
        for test_year in range(syear, eyear):
            train_dataset = TrainDataset(test_year)
            # 加载测试集
            test_dataset = TestDataset(test_year, test_year, train_dataset)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
            # 读取模型
            model = LSTM_CNN(train_dataset.shape)
            model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_1991-2019年模型(除{test_year}年).pth"
            model.load_state_dict(torch.load(model_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 取训练集中的最值用于反归一化
            tensor_min = train_dataset.min
            tensor_max = train_dataset.max

            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    # 反归一化
                    outputs = OtherUtils.min_max_denormalization(outputs, tensor_min, tensor_max)
                    inputs = OtherUtils.min_max_denormalization(inputs, tensor_min, tensor_max)
                    labels = OtherUtils.min_max_denormalization(labels, tensor_min, tensor_max)

                    print(f"{test_year}年{MONTHS[i]}月")
                    print("订正前mse", utils.OtherUtils.mse(inputs, labels))
                    print("订正后mse", utils.OtherUtils.mse(outputs, labels))

                    # 添加当前月全部年份的输入、输出、标签用于计算评价指标
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

                    # 保存结果图
                    fig_title = f"{TIME}-{BASIN}-{AREA}-{test_year}年{MONTHS[i]}月"
                    result_path = RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}"
                    utils.PaintUtils.paint_result(fig_title, result_path, inputs[0][i], outputs[0][i], labels[0][i])

        # TCC相关
        tcc_path = EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
        corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)  # 订正后与真实值
        case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)  # 订正前与真实值
        tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
        os.makedirs(tcc_path, exist_ok=True)
        tcc_img.savefig(rf"{tcc_path}/{AREA}_91-19年{MONTHS[i]}月")
        print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[i]}月")
        tcc_img.close()

        # ACC相关
        acc_path = EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
        corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses, False)
        case_acc = OtherUtils.cal_ACC(test_cases, test_obses, False)
        acc_img = PaintUtils.paint_ACC(range(syear, eyear), case_acc, corr_acc)
        os.makedirs(acc_path, exist_ok=True)
        acc_img.savefig(rf"{acc_path}/{AREA}_91-19年{MONTHS[i]}月")
        print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[i]}月")
        acc_img.close()


if __name__ == '__main__':
    plt.rcParams['axes.unicode_minus'] = False
    test()
