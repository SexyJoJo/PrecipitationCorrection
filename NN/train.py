"""模型搭建与训练"""
import os.path
import random
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
from NN_CONST import *
from models import *


# 日志初始化
logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s  %(message)s')

# 路径初始化
CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)
SHAPE = np.ones((1, 5, 3, 3)).shape
MONTHS = utils.OtherUtils.get_predict_months(DATE, 1)


class TrainDataset(Dataset):
    def __init__(self, JUMP_YEAR):
        super().__init__()
        # 读取原始数据
        self.obs_data = utils.ObsParser.get_many_2d_pravg(
            OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, MONTHS, JUMP_YEAR
        )
        self.case_data = utils.CaseParser.get_many_2d_pravg(
            CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR
        )

        if USE_ANOMALY:
            # 原始数据转距平
            self.case_avg = np.mean(self.case_data, axis=0)
            self.obs_avg = np.mean(self.obs_data, axis=0)
            self.case_data = self.case_data - self.case_avg
            self.obs_data = self.obs_data - self.obs_avg

        # 数据维度重新组织为n*3*3
        self.obs_data, self.valid_indexes = utils.ObsParser.organize_input(self.obs_data)
        self.case_data = utils.CaseParser.organize_input(self.case_data, self.valid_indexes)

        # 数据归一化
        self.case_data = torch.Tensor(self.case_data)
        self.obs_data = torch.Tensor(self.obs_data)
        # all_data = torch.cat([self.case_data, self.obs_data], 0)
        self.data_min = torch.min(self.case_data)
        self.data_max = torch.max(self.case_data)
        self.case_data = utils.OtherUtils.min_max_normalization(self.case_data, self.data_min, self.data_max)
        self.obs_data = utils.OtherUtils.min_max_normalization(self.obs_data, self.data_min, self.data_max)
        # self.len = self.case_data.shape

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


class TestDataset(Dataset):
    def __init__(self, TEST_START_YEAR, TEST_END_YEAR, train_dataset):
        super().__init__()
        # 读取原始数据
        self.obs_data = utils.ObsParser.get_many_2d_pravg(
            OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, MONTHS)
        self.case_data = utils.CaseParser.get_many_2d_pravg(
            CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA)

        if USE_ANOMALY:
            # 数据转距平
            self.case_data = self.case_data - train_dataset.case_avg
            self.obs_data = self.obs_data - train_dataset.obs_avg

        # 数据维度重新组织为n*3*3
        self.obs_data, self.valid_indexes = utils.ObsParser.organize_input(self.obs_data)
        self.case_data = utils.CaseParser.organize_input(self.case_data, self.valid_indexes)

        # 数据归一化
        self.case_data = torch.Tensor(self.case_data)
        self.obs_data = torch.Tensor(self.obs_data)
        self.case_data = utils.OtherUtils.min_max_normalization(
            self.case_data, train_dataset.data_min, train_dataset.data_max)
        self.obs_data = utils.OtherUtils.min_max_normalization(
            self.obs_data, train_dataset.data_min, train_dataset.data_max)

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(test_year):
    total_training_loss_list = []  # 用于绘制损失趋势图
    total_test_loss_list = []
    # tensor_min, tensor_max = get_minmax(test_year)

    for epoch in range(EPOCH):
        # 模型训练
        total_training_loss = 0.
        for i, data in enumerate(train_dataloader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            training_loss = criterion(outputs, target)
            total_training_loss += training_loss.item()
            print(f"epoch:{epoch}  i:{i}   training_loss:{training_loss.item()}  "
                  f"total_training_loss:{total_training_loss}")
            # logging.debug(f"epoch:{epoch}  i:{i}   training_loss:{training_loss.item()}  "
            #               f"total_training_loss:{total_training_loss}")
            training_loss.backward()
            optimizer.step()
        total_training_loss_list.append(total_training_loss)

        # 模型验证
        total_test_loss = 0.
        with torch.no_grad():
            for data in test_dataloader:
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                total_test_loss += test_loss.item()
                print(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
                      f"total_testing_loss:{total_test_loss}")
                # logging.debug(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
                #               f"total_testing_loss:{total_test_loss}")
            total_test_loss_list.append(total_test_loss)

    # 绘制损失
    plt.plot(list(range(EPOCH)), total_training_loss_list, label="total training loss")
    plt.plot(list(range(EPOCH)), total_test_loss_list, label="total test loss")
    # plt.ylim([0, 0.005])
    plt.legend()
    os.makedirs(LOSS_PATH, exist_ok=True)
    plt.savefig(LOSS_PATH + rf"/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年损失(除{test_year}).png")
    plt.close()


if __name__ == '__main__':
    setup_seed(24)
    # 所有年份中选一年作为测试集，其他年份作为训练集，以不同的训练集循环训练多个模型
    for TEST_YEAR in range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1):
        # 初始化模型与数据集
        model = LSTM(input_size=45, hidden_size=64, num_layers=1, output_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # 加载训练集
        train_dataset = TrainDataset(TEST_YEAR)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
        # 加载测试集 用于查看损失
        test_dataset = TestDataset(TEST_YEAR, TEST_YEAR, train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        # # 取训练集中的最值用于反归一化
        # train_case_data = torch.Tensor(utils.CaseParser.get_many_2d_pravg(
        #     CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, TEST_YEAR, USE_ANOMALY, case_avg))
        # train_obs_data = torch.Tensor(utils.ObsParser.get_many_2d_pravg(
        #     OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, MONTHS, TEST_YEAR, USE_ANOMALY, case_avg))
        # all_train_data = torch.cat([train_case_data, train_obs_data], 0)
        # tensor_min = torch.min(all_train_data)
        # tensor_max = torch.max(all_train_data)

        # 开始训练
        logging.debug(f"开始训练1991-2019年模型({TEST_YEAR}年除外)")
        print(f"开始训练1991-2019年模型({TEST_YEAR}年除外)")
        start = datetime.now()
        train(TEST_YEAR)
        end = datetime.now()
        logging.debug(f"模型训练完成,耗时:{end - start}")
        print("模型训练完成,耗时:", end - start)

        os.makedirs(MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
        model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年模型(除{TEST_YEAR}年).pth"
        torch.save(model.state_dict(), model_path)
        print("保存模型文件:", model_path)

    # data = torch.rand((20, 5, 43, 39))
    # output = model(data)
    # print(output.shape)
