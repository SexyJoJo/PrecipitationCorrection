"""模型搭建与训练"""
import os.path
import random
from datetime import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import utils
from NN_CONST import *


class TestDataset(Dataset):
    def __init__(self, TEST_START_YEAR, TEST_END_YEAR):
        super().__init__()
        self.case_data = torch.Tensor(utils.CaseParser.get_many_2d_pravg(
            CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA))
        self.obs_data = torch.Tensor(utils.ObsParser.get_many_2d_pravg(
            OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, MONTHS))

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


class TrainDataset(Dataset):
    def __init__(self, JUMP_YEAR):
        super().__init__()
        # 读取原始数据
        self.case_data = torch.Tensor(utils.CaseParser.get_many_2d_pravg(
            CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR, DATA_ENHANCE
        ))
        self.shape = self.case_data.shape
        self.months = utils.OtherUtils.get_predict_months(DATE, self.shape[1])
        self.obs_data = torch.Tensor(utils.ObsParser.get_many_2d_pravg(
            OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, self.months, JUMP_YEAR, DATA_ENHANCE
        ))

        # 数据转距平
        self.case_avg = np.mean(self.case_data.numpy(), axis=0)
        self.obs_avg = np.mean(self.obs_data.numpy(), axis=0)
        self.case_data = self.case_data - self.case_avg
        self.obs_data = self.obs_data - self.obs_avg

        # 数据归一化
        all_data = torch.cat([self.case_data, self.obs_data], 0)
        self.tensor_min = torch.min(all_data)
        self.tensor_max = torch.max(all_data)
        self.case_data = utils.OtherUtils.min_max_normalization(self.case_data, self.tensor_min, self.tensor_max)
        self.obs_data = utils.OtherUtils.min_max_normalization(self.obs_data, self.tensor_min, self.tensor_max)

        # self.len = self.case_data.shape

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        # 定义CNN层
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=2880, hidden_size=64, num_layers=2, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(in_features=64, out_features=5 * SHAPE[2] * SHAPE[3])

    def forward(self, x):
        # CNN层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # 把输出的特征图转化为LSTM层的输入
        x = x.view(-1, 32 * x.shape[2] * x.shape[3])

        # LSTM层
        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取LSTM层最后一个时间步的输出

        # 全连接层
        x = self.fc(x)
        x = x.view(-1, 5, SHAPE[2], SHAPE[3])

        return x


# 日志初始化
logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s  %(message)s')
# 路径初始化
CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)
SHAPE = TrainDataset(1991).shape
MONTHS = utils.OtherUtils.get_predict_months(DATE, 5)


# class NN(nn.Module):
#     # 当前维度（43*39）针对金沙江流域， 其他流域需要更改维度
#     def __init__(self):
#         super(NN, self).__init__()
#
#         # 输入[batch, 月份, 43, 39]
#         self.lstm = nn.LSTM(
#             input_size=SHAPE[2] * SHAPE[3],
#             hidden_size=SHAPE[2] * SHAPE[3],
#             bidirectional=True,
#             batch_first=True
#         )
#         self.layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(SHAPE[2] * SHAPE[3] * 2, SHAPE[2] * SHAPE[3])
#         )
#
#         # 第一层卷积
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(  # 定义1个2维的卷积核
#                 in_channels=5,  # 输入通道的个数（单个case预报的月份个数）
#                 out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
#                 kernel_size=(3, 3),  # 卷积核的大小
#                 stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
#                 padding=1,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
#             ),
#             nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2)  # 经过最大值池化 输出传入下一个卷积
#
#             nn.Conv2d(
#                 in_channels=16,  # 输入个数与上层输出一致
#                 out_channels=32,
#                 kernel_size=(3, 3),
#                 stride=(1, 1),
#                 padding=1
#             ),
#             nn.ReLU(),
#
#             nn.Conv2d(
#                 in_channels=32,  # 输入个数与上层输出一致
#                 out_channels=64,
#                 kernel_size=(3, 3),
#                 stride=(1, 1),
#                 padding=1
#             ),
#             nn.ReLU(),
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=64,  # 输入个数与上层输出一致
#                 out_channels=SHAPE[1],
#                 kernel_size=(3, 3),
#                 stride=(1, 1),
#                 padding=1
#             ),
#         )
#
#     # def attention_net(self, lstm_output, final_state):
#     #     hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
#     #     attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
#     #     soft_attn_weights = F.softmax(attn_weights, 1)
#     #     # context : [batch_size, n_hidden * num_directions(=2)]
#     #     context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
#     #
#     #     return context, soft_attn_weights
#
#     def forward(self, x):
#         x = x.flatten(-2, -1)
#         x, (final_hidden_state, final_cell_state) = self.lstm(x)
#         # x, attention = self.attention_net(x, final_hidden_state)
#         # x = x.transpose(0, 1)
#         x = self.layer(x)
#         x = x.unflatten(-1, (SHAPE[2], SHAPE[3]))
#         x = x[:, None]
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(test_year):
    total_training_loss_list = []  # 用于绘制损失趋势图
    total_test_loss_list = []
    tensor_min, tensor_max = train_dataset.tensor_min, train_dataset.tensor_max

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
            if (i + 1) % (28 / BATCH_SIZE) == 0:
                total_training_loss_list.append(total_training_loss)
            print(f"epoch:{epoch}  i:{i}   training_loss:{training_loss.item()}  "
                  f"total_training_loss:{total_training_loss}")
            logging.debug(f"epoch:{epoch}  i:{i}   training_loss:{training_loss.item()}  "
                          f"total_training_loss:{total_training_loss}")
            training_loss.backward()
            optimizer.step()

        # 模型验证
        total_test_loss = 0.
        with torch.no_grad():
            for data in test_dataloader:
                data[0] = utils.OtherUtils.min_max_normalization(data[0], tensor_min, tensor_max)
                data[1] = utils.OtherUtils.min_max_normalization(data[1], tensor_min, tensor_max)
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                total_test_loss += test_loss.item()
                total_test_loss_list.append(total_test_loss)
                print(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
                      f"total_testing_loss:{total_test_loss}")
                logging.debug(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
                              f"total_testing_loss:{total_test_loss}")

    # 绘制损失
    plt.plot(list(range(EPOCH)), total_training_loss_list, label="total training loss")
    plt.plot(list(range(EPOCH)), total_test_loss_list, label="total test loss")
    # plt.ylim([0, 0.005])
    plt.legend()
    plt.savefig(rf"./loss/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年损失(除{test_year}).png")
    plt.close()


if __name__ == '__main__':
    setup_seed(24)
    # 所有年份中选一年作为测试集，其他年份作为训练集，以不同的训练集循环训练多个模型
    for TEST_YEAR in range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1):
        # 加载训练集
        train_dataset = TrainDataset(TEST_YEAR)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
        SHAPE = train_dataset.shape
        MONTHS = train_dataset.months
        # 加载测试集 用于查看损失
        test_dataset = TestDataset(TEST_YEAR, TEST_YEAR)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

        # 初始化模型与数据集
        model = CNN_LSTM()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 开始训练
        logging.debug(f"开始训练1991-2019年模型({TEST_YEAR}年除外)")
        print(f"开始训练1991-2019年模型({TEST_YEAR}年除外)")
        start = datetime.now()
        train(TEST_YEAR)
        end = datetime.now()
        logging.debug(f"模型训练完成,耗时:{end - start}")
        print("模型训练完成,耗时:", end - start)

        os.makedirs(rf"距平models/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
        model_path = rf"距平models/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年模型(除{TEST_YEAR}年).pth"
        torch.save(model.state_dict(), model_path)
        print("保存模型文件:", model_path)

    # data = torch.rand((20, 5, 43, 39))
    # output = model(data)
    # print(output.shape)
