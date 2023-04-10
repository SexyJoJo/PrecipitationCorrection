import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM_CNN(nn.Module):
    def __init__(self, shape):
        super(LSTM_CNN, self).__init__()
        self.shape = shape

        # 输入[batch, 5,43,39]
        self.lstm = nn.LSTM(
            input_size=shape[2] * shape[3],
            hidden_size=shape[2] * shape[3],
            bidirectional=True,
            batch_first=True
        )
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(shape[2] * shape[3] * 2, shape[2] * shape[3])
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
                out_channels=shape[1],
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
        x = x.unflatten(-1, (self.shape[2], self.shape[3]))
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ANN(nn.Module):
    def __init__(self, shape):
        super(ANN, self).__init__()

        self.fc1 = nn.Linear(shape[1], 4, bias=True)
        self.fc2 = nn.Linear(4, 4, bias=True)
        self.fc3 = nn.Linear(4, 1, bias=True)
        self.tanh = nn.Tanh()  # 双曲正切激活函数
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x
