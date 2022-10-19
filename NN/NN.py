"""CNN + BiLSTM + Attention"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import utils
from NN_CONST import *


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # 输入[batch, 5,43,39]
        self.lstm = nn.LSTM(input_size=43 * 39, hidden_size=43 * 39, bidirectional=True, batch_first=True)
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(43 * 39 * 2, 43 * 39)
        )

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 定义1个2维的卷积核
                in_channels=1,  # 输入通道的个数（单个case预报的月份个数）
                out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
                kernel_size=(3, 3),  # 卷积核的大小
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
                padding=1,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)  # 经过最大值池化 输出传入下一个卷积

            nn.Conv2d(
                in_channels=16,  # 输入个数与上层输出一致
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,  # 输入个数与上层输出一致
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # 输入个数与上层输出一致
                out_channels=5,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
        )

    def attention_net(self, lstm_output, final_state):
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, x):
        x = x.flatten(-2, -1)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        x, attention = self.attention_net(x, final_hidden_state)
        # x = x.transpose(0, 1)
        x = self.layer(x)
        x = x.unflatten(-1, (43, 39))
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA))
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, MONTHS))
        # self.len = self.case_data.shape

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA))
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, MONTHS))

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


def train():
    for epoch in range(EPOCH):
        for i, data in enumerate(train_dataloader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            print(epoch, i, loss.item())
            loss.backward()
            optimizer.step()


# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_dataloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('accuracy on test set: %d %% ' % (100*correct/total))
#     return correct/total


# 测试模型代码
# def test_model():
#     input = torch.ones((5, 6, 43, 39))  # 5个图片， 6个通道
#     model = NN()
#     output = model(input)
#     print(output)
#     test_input = torch.ones(1, )


model = NN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 加载数据集
train_dataset = TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset)
test_dataset = TestDataset()
test_dataloader = DataLoader(dataset=test_dataset)

if __name__ == '__main__':
    train()
    # data = torch.rand((20, 5, 43, 39))
    # output = model(data)
    # print(output.shape)
