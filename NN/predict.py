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
    utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_START_YEAR, AREA)).shape
# MONTHS = utils.OtherUtils.get_predict_months(DATE, SHAPE[1])
MONTHS = utils.OtherUtils.get_predict_months(DATE, 1)


torch.manual_seed(42)


def test():
    na_list, valid_grids = utils.ObsParser.get_na_index(OBS_DIR, AREA)
    syear, eyear = 1991, 2020

    if DATA_FORMAT == 'grid':
        for m in range(len(MONTHS)):
            corr_cases, test_cases, test_obses = [], [], []
            anomaly_test_cases, anomaly_test_obses = [], []

            for test_year in range(syear, eyear):
                # 输出数组初始化
                inputs_map = np.zeros([SHAPE[2], SHAPE[3]]) + np.nan
                outputs_map = np.zeros([SHAPE[2], SHAPE[3]]) + np.nan
                labels_map = np.zeros([SHAPE[2], SHAPE[3]]) + np.nan
                # 加载训练集
                train_dataset = TrainDataset(test_year)
                # 加载测试集
                test_dataset = TestDataset(test_year, test_year, train_dataset)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
                # 读取模型
                # model = LSTM_CNN(train_dataset.shape)
                # model = ANN(train_dataset.shape)
                model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_1991-2019年模型(除{test_year}年).pth"
                model = torch.load(model_path)
                # model.load_state_dict(torch.load(model_path))
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 预测test_year m月的各格点值
                with torch.no_grad():
                    for cnt, data in enumerate(test_dataloader):
                        # 预测
                        inputs, labels = data
                        # inputs, labels = inputs.to(device), labels.to(device)
                        # inputs, labels = test_dataset.case_data, test_dataset.obs_data
                        outputs = model(inputs)
                        # 反归一化
                        if NORMALIZATION == 'minmax':
                            tensor_min = train_dataset.min
                            tensor_max = train_dataset.max
                            outputs = OtherUtils.min_max_denormalization(outputs, tensor_min, tensor_max)
                            inputs = OtherUtils.min_max_denormalization(inputs, tensor_min, tensor_max)
                            labels = OtherUtils.min_max_denormalization(labels, tensor_min, tensor_max)
                        elif NORMALIZATION == 'zscore':
                            outputs = OtherUtils.zscore_denormalization(
                                outputs, train_dataset.obs_means, train_dataset.obs_stds, DATA_FORMAT)
                            inputs = OtherUtils.zscore_denormalization(
                                inputs, train_dataset.case_means, train_dataset.case_stds, DATA_FORMAT)
                            labels = OtherUtils.zscore_denormalization(
                                labels, train_dataset.obs_means, train_dataset.obs_stds, DATA_FORMAT)
                        # 重组结果
                        a, b = valid_grids[cnt]
                        inputs_map[a][b] = inputs[0][m]
                        outputs_map[a][b] = outputs[0][m]
                        labels_map[a][b] = labels[0][m]

                # inputs_map = torch.Tensor(inputs_map)
                # outputs_map = torch.Tensor(outputs_map)
                # labels_map = torch.Tensor(labels_map)

                # 绘结果图
                print(f"{test_year}年{MONTHS[m]}月")
                print("订正前mse", utils.OtherUtils.cal_mse(inputs_map, labels_map))
                print("订正后mse", utils.OtherUtils.cal_mse(outputs_map, labels_map))

                test_cases.append(inputs_map)
                corr_cases.append(outputs_map)
                test_obses.append(labels_map)
                anomaly_test_cases.append(inputs_map - train_dataset.case_grids_means[m])
                anomaly_test_obses.append(labels_map - train_dataset.obs_grids_means[m])

                # 保存结果图
                fig_title = f"{TIME}-{BASIN}-{AREA}-{test_year}年{MONTHS[m]}月"
                result_path = RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}"
                utils.PaintUtils.paint_result(fig_title, result_path,
                                              torch.Tensor(inputs_map),
                                              torch.Tensor(outputs_map),
                                              torch.Tensor(labels_map))

            anomaly_corr_cases = np.array(corr_cases) - np.mean(np.array(corr_cases), axis=0)
            # TCC相关
            tcc_path = EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
            corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)  # 订正后与真实值
            case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)  # 订正前与真实值
            tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
            os.makedirs(tcc_path, exist_ok=True)
            tcc_img.savefig(rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月")
            print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月")
            tcc_img.close()
            corr_tcc = OtherUtils.cal_TCC(anomaly_corr_cases, anomaly_test_obses)  # 订正后与真实值
            case_tcc = OtherUtils.cal_TCC(anomaly_test_cases, anomaly_test_obses)  # 订正前与真实值
            tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
            tcc_img.savefig(rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            tcc_img.close()

            # ACC相关
            acc_path = EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
            corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses, False)
            case_acc = OtherUtils.cal_ACC(test_cases, test_obses, False)
            acc_img = PaintUtils.paint_ACC(range(syear, eyear), case_acc, corr_acc)
            os.makedirs(acc_path, exist_ok=True)
            acc_img.savefig(rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月")
            print("acc已保存", rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月")
            acc_img.close()
            corr_acc = OtherUtils.cal_ACC(anomaly_corr_cases, anomaly_test_obses, False)
            case_acc = OtherUtils.cal_ACC(anomaly_test_cases, anomaly_test_obses, False)
            acc_img = PaintUtils.paint_ACC(range(syear, eyear), case_acc, corr_acc)
            acc_img.savefig(rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            print("acc已保存", rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            acc_img.close()

    elif DATA_FORMAT == 'map':
        for m in range(len(MONTHS)):
            corr_cases, test_cases, test_obses = [], [], []
            anomaly_test_cases, anomaly_test_obses = [], []
            for test_year in range(syear, eyear):
                train_dataset = TrainDataset(test_year)
                # 加载测试集
                test_dataset = TestDataset(test_year, test_year, train_dataset)
                test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
                # 读取模型
                # model = LSTM_CNN(train_dataset.shape)
                model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_1991-2019年模型(除{test_year}年).pth"
                # model.load_state_dict(torch.load(model_path))
                model = torch.load(model_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                with torch.no_grad():
                    for cnt, data in enumerate(test_dataloader):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)

                        # 反归一化
                        if NORMALIZATION == 'minmax':
                            tensor_min = train_dataset.min
                            tensor_max = train_dataset.max
                            outputs = OtherUtils.min_max_denormalization(outputs, tensor_min, tensor_max)
                            inputs = OtherUtils.min_max_denormalization(inputs, tensor_min, tensor_max)
                            labels = OtherUtils.min_max_denormalization(labels, tensor_min, tensor_max)
                        elif NORMALIZATION == 'zscore':
                            outputs = OtherUtils.zscore_denormalization(
                                outputs, train_dataset.obs_means, train_dataset.obs_stds, DATA_FORMAT)
                            inputs = OtherUtils.zscore_denormalization(
                                inputs, train_dataset.case_means, train_dataset.case_stds, DATA_FORMAT)
                            labels = OtherUtils.zscore_denormalization(
                                labels, train_dataset.obs_means, train_dataset.obs_stds, DATA_FORMAT)

                        print(f"{test_year}年{MONTHS[m]}月")
                        print("订正前mse", utils.OtherUtils.cal_mse(inputs[0][m], labels[0][m]))
                        print("订正后mse", utils.OtherUtils.cal_mse(outputs[0][m], labels[0][m]))

                        # 添加当前月全部年份的输入、输出、标签用于计算评价指标
                        test_case = inputs[0][m].cpu().numpy()
                        corr_case = outputs[0][m].cpu().numpy()
                        test_obs = labels[0][m].cpu().numpy()
                        for a, b in na_list:
                            test_case[a][b] = np.nan
                            corr_case[a][b] = np.nan
                            test_obs[a][b] = np.nan
                        test_cases.append(test_case)
                        corr_cases.append(corr_case)
                        test_obses.append(test_obs)
                        anomaly_test_cases.append(test_case - train_dataset.case_grids_means[m])
                        anomaly_test_obses.append(test_obs - train_dataset.obs_grids_means[m])

                        # 保存结果图
                        fig_title = f"{TIME}-{BASIN}-{AREA}-{test_year}年{MONTHS[m]}月"
                        result_path = RESULT_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}"
                        utils.PaintUtils.paint_result(fig_title, result_path, inputs[0][m], outputs[0][m], labels[0][m])

            anomaly_corr_cases = np.array(corr_cases) - np.mean(np.array(corr_cases), axis=0)
            # TCC相关
            tcc_path = EVALUATE_PATH + rf"/TCC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
            corr_tcc = OtherUtils.cal_TCC(corr_cases, test_obses)  # 订正后与真实值
            case_tcc = OtherUtils.cal_TCC(test_cases, test_obses)  # 订正前与真实值
            tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
            os.makedirs(tcc_path, exist_ok=True)
            tcc_img.savefig(rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月")
            print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月")
            tcc_img.close()
            corr_tcc = OtherUtils.cal_TCC(anomaly_corr_cases, anomaly_test_obses)  # 订正后与真实值
            case_tcc = OtherUtils.cal_TCC(anomaly_test_cases, anomaly_test_obses)  # 订正前与真实值
            tcc_img = PaintUtils.paint_TCC(case_tcc, corr_tcc)
            tcc_img.savefig(rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            print("tcc已保存", rf"{tcc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            tcc_img.close()

            # ACC相关
            acc_path = EVALUATE_PATH + rf"/ACC/{DATE}/{CASE_NUM}/{TIME}/{BASIN}"
            corr_acc = OtherUtils.cal_ACC(corr_cases, test_obses, False)
            case_acc = OtherUtils.cal_ACC(test_cases, test_obses, False)
            acc_img = PaintUtils.paint_ACC(range(syear, eyear), case_acc, corr_acc)
            os.makedirs(acc_path, exist_ok=True)
            acc_img.savefig(rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月")
            print("acc已保存", rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月")
            acc_img.close()
            corr_acc = OtherUtils.cal_ACC(anomaly_corr_cases, anomaly_test_obses, False)
            case_acc = OtherUtils.cal_ACC(anomaly_test_cases, anomaly_test_obses, False)
            acc_img = PaintUtils.paint_ACC(range(syear, eyear), case_acc, corr_acc)
            acc_img.savefig(rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            print("acc已保存", rf"{acc_path}/{AREA}_91-19年{MONTHS[m]}月-距平")
            acc_img.close()



if __name__ == '__main__':
    plt.rcParams['axes.unicode_minus'] = False
    test()
