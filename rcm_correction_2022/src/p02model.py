#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   PrecipitationCorrection_qiaogit
@file    :   p02model.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-1-16 19:19
@description:
模型定义
'''

# here put the import lib
import torch
import torch.nn as nn


# 模型定义类
# 使用LSTM构造模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirect=True):
        super(LSTMModel, self).__init__()
        self.__hidden_dim = hidden_dim  # 隐藏层节点数量
        self.__num_layers = num_layers  # 隐藏层数量
        self.__bidirect = bidirect
        if bidirect:
            self.__num_direct = 2
        else:
            self.__num_direct = 1

        self.__attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)

        self.__lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirect)
        self.__fc = nn.Linear(hidden_dim * self.__num_direct, output_dim)

    def forward(self, x):
        batch_size_temp = x.size(0)

        h0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)
        c0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)

        output, (hn, cn) = self.__lstm(x, (h0, c0))
        # output = output.view(batch_size_temp, -1)
        output = self.__fc(output)
        return output[:, -1]

    def __str__(self):
        s = f'lstm-l{self.__num_layers}h{self.__hidden_dim}d{self.__num_direct}'
        return s


# 使用经典的ANN建模
class ANNModel(torch.nn.Module):
    def __init__(self, input_dims=1, output_dims=1, layers=1,
                 hiddens=128, activation='tanh',
                 batchnormflag=False, droupoutflag=False):
        super(ANNModel, self).__init__()

        self.__hiddens = hiddens
        self.__layers = layers
        self.__activation = activation.lower()
        self.__batchnormflag = batchnormflag
        self.__droupoutflag = droupoutflag

        net = nn.Sequential()
        net.add_module(r'inputlayer',
                       nn.Linear(in_features=input_dims,
                                 out_features=self.__hiddens, bias=True))
        if self.__activation == 'tanh':
            net.add_module(r'inputact', nn.Tanh())
        elif self.__activation == 'relu':
            net.add_module(r'inputact', nn.ReLU())
        else:
            net.add_module(r'inputact', nn.Sigmoid())

        for i in range(self.__layers):
            net.add_module(rf'hidlayer-{i + 1}',
                           nn.Linear(in_features=self.__hiddens,
                                     out_features=self.__hiddens, bias=True))

            if self.__batchnormflag:
                net.add_module(rf'bn-{i + 1}', nn.BatchNorm1d(self.__hiddens))

            if self.__droupoutflag:
                net.add_module(rf'drop-{i + 1}', nn.Dropout())

            if self.__activation == 'tanh':
                net.add_module(rf'hidact-{i + 1}', nn.Tanh())
            elif self.__activation == 'relu':
                net.add_module(rf'hidact-{i + 1}', nn.ReLU())
            else:
                net.add_module(rf'hidact-{i + 1}', nn.Sigmoid())

        net.add_module(r'outputlayer',
                       nn.Linear(in_features=self.__hiddens,
                                 out_features=output_dims, bias=True))
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x

    def __str__(self):
        s = f'ann-l{self.__layers}h{self.__hiddens}-{self.__activation}' \
            f'-bn{1 if self.__batchnormflag else 0}-dp{1 if self.__droupoutflag else 0}'
        return s


# 使用经典ANN和LSTM混合建模
# 数据方式：前两列为经纬度；后面若干列为输入数据，也可以视为序列数据
class ANN_LSTM_Model(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, num_layers=1,
                 hidden_dim=128, activation='tanh',
                 batchnormflag=False, droupoutflag=False,
                 bidirect=True):
        super(ANN_LSTM_Model, self).__init__()

        # 构造LSTM模块
        self.__hidden_dim = hidden_dim  # 隐藏层节点数量
        self.__num_layers = num_layers  # 隐藏层数量
        self.__bidirect = bidirect
        if bidirect:
            self.__num_direct = 2
        else:
            self.__num_direct = 1

        self.__attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)

        self.__lstm = nn.LSTM(input_dim - 2, hidden_dim, num_layers, batch_first=True, bidirectional=bidirect)
        # self.__fc = nn.Linear(hidden_dim * self.__num_direct, output_dim)

        # 构造普通ANN模块
        self.__activation = activation.lower()
        self.__batchnormflag = batchnormflag
        self.__droupoutflag = droupoutflag

        ann_net = nn.Sequential()
        ann_net.add_module(r'inputlayer',
                           nn.Linear(in_features=input_dim,
                                     out_features=self.__hidden_dim, bias=True))
        if self.__activation == 'tanh':
            ann_net.add_module(r'inputact', nn.Tanh())
        elif self.__activation == 'relu':
            ann_net.add_module(r'inputact', nn.ReLU())
        else:
            ann_net.add_module(r'inputact', nn.Sigmoid())

        for i in range(self.__num_layers):
            ann_net.add_module(rf'hidlayer-{i + 1}',
                               nn.Linear(in_features=self.__hidden_dim,
                                         out_features=self.__hidden_dim, bias=True))

            if self.__batchnormflag:
                ann_net.add_module(rf'bn-{i + 1}', nn.BatchNorm1d(self.__hidden_dim))

            if self.__droupoutflag:
                ann_net.add_module(rf'drop-{i + 1}', nn.Dropout())

            if self.__activation == 'tanh':
                ann_net.add_module(rf'hidact-{i + 1}', nn.Tanh())
            elif self.__activation == 'relu':
                ann_net.add_module(rf'hidact-{i + 1}', nn.ReLU())
            else:
                ann_net.add_module(rf'hidact-{i + 1}', nn.Sigmoid())

        # ann_net.add_module(r'outputlayer',
        #                nn.Linear(in_features=self.__hidden_dim,
        #                          out_features=output_dim, bias=True))
        self.__ann_net = ann_net

        self.__fc = nn.Linear(self.__hidden_dim * self.__num_direct + self.__hidden_dim,
                              output_dim)

    def forward(self, x):
        batch_size_temp = x.size(0)
        ann_x = self.__ann_net(x)

        h0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)
        c0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)

        lstm_x = x[:, 2:]
        lstm_x = lstm_x.view(len(lstm_x), 1, -1)  # 增加一个len维度，以对应lstm
        output_lstm, (hn, cn) = self.__lstm(lstm_x, (h0, c0))
        output_lstm = output_lstm.view(batch_size_temp, -1)

        full_x = torch.cat([ann_x, output_lstm], dim=1)
        output = self.__fc(full_x)
        return output

    def __str__(self):
        s = f'ann-lstm-l{self.__num_layers}h{self.__hidden_dim}d{self.__num_direct}'
        return s


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirect=True):
        super(Model, self).__init__()
        self.__hidden_dim = hidden_dim  # 隐藏层节点数量
        self.__num_layers = num_layers  # 隐藏层数量
        self.__bidirect = bidirect
        if bidirect:
            self.__num_direct = 2
        else:
            self.__num_direct = 1

        self.__attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)

        self.__lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirect)
        self.__fc = nn.Linear(hidden_dim * self.__num_direct, output_dim)

    def forward(self, x):
        batch_size_temp = x.size(0)

        h0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)
        c0 = torch.zeros(self.__num_layers * self.__num_direct, batch_size_temp, self.__hidden_dim)

        output, (hn, cn) = self.__lstm(x, (h0, c0))
        # output = output.view(batch_size_temp, -1)
        output = self.__fc(output)
        return output[:, -1]

    def __str__(self):
        s = f'lstm-l{self.__num_layers}h{self.__hidden_dim}d{self.__num_direct}'
        return s


# 测试模型
def z01test():
    # model = Model(5, 64, 1, 1, bidirect=True)
    # x = torch.zeros([64, 5], dtype=torch.float)
    # x = x.view(len(x), 1, -1)   # 增加序列的维度
    # y = model(x)
    # print(y.shape)

    # model = ANNModel(input_dims=8, output_dims=1, layers=1,
    #                  hiddens=16, activation='tanh',
    #                  batchnormflag=False, droupoutflag=False)
    # x = torch.zeros([64, 8], dtype=torch.float)
    # y = model(x)
    # print(y.shape)

    model = ANN_LSTM_Model(input_dim=8, output_dim=1, num_layers=1,
                           hidden_dim=16, activation='tanh',
                           batchnormflag=False, droupoutflag=False)
    x = torch.zeros([64, 8], dtype=torch.float)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    z01test()
    pass
