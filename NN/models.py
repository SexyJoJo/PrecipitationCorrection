import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM_CNN(nn.Module):
    # 当前维度（43*39）针对金沙江流域， 其他流域需要更改维度
    def __init__(self):
        super(LSTM_CNN, self).__init__()

        # 输入[batch, 月份, 43, 39]
        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=9,
            bidirectional=True,
            batch_first=True
        )
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(9 * 2, 9)
        )

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 定义1个2维的卷积核
                in_channels=5,  # 输入通道的个数（单个case预报的月份个数）
                out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
                kernel_size=(3, 3),  # 卷积核的大小
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
                padding=1,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
            ),
            nn.ReLU(),
        )

        self.fc = nn.Linear(16 * 3 * 3, 5 * 3 * 3)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=16,  # 输入个数与上层输出一致
        #         out_channels=SHAPE[1],
        #         kernel_size=(3, 3),
        #         stride=(1, 1),
        #         padding=1
        #     ),
        # )

    # def attention_net(self, lstm_output, final_state):
    #     hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
    #     attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
    #     soft_attn_weights = F.softmax(attn_weights, 1)
    #     # context : [batch_size, n_hidden * num_directions(=2)]
    #     context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    #
    #     return context, soft_attn_weights

    def forward(self, x):
        batch_size = x.size(0)
        x = x.flatten(-2, -1)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        # x, attention = self.attention_net(x, final_hidden_state)
        # x = x.transpose(0, 1)
        x = self.layer(x)
        x = x.unflatten(-1, (3, 3))
        x = self.conv1(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = x.unflatten(-1, (5, 3, 3))
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(len(x), 5, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.unflatten(-1, (1, 1, 1))
        return x


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.fc1 = nn.Linear(45, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 64)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(64, 64)  # 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(64, 1)  # 第三个隐藏层到输出层
        self.tanh = nn.Tanh()  # 双曲正切激活函数
        self.dropout = nn.Dropout(p=1)  # Dropout概率为1

    def forward(self, x):
        x = x.flatten(-2, -1)
        x = x.flatten(-2, -1)

        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = x.unflatten(-1, (1, 1, 1))
        return x
