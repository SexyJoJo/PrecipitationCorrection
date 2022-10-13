"""CNN + BiLSTM + Attention"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[6,43,39]
            nn.Conv2d(  # 定义1个2维的卷积核
                in_channels=6,  # 输入通道的个数（单个case预报的月份个数）
                out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
                kernel_size=(3, 3),  # 卷积核的大小
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
                padding=1,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
            ),
            # 经过卷积层 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过最大值池化 输出传入下一个卷积
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 输入个数与上层输出一致
                out_channels=32,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            # 经过卷积 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出传入输出层
        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 输入个数与上层输出一致
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            # 经过卷积 输出[64, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出传入输出层
        )

        # 输出层（全连接）
        # self.output = nn.Linear(in_features=, out_features=1)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=64 * 5 * 4, hidden_size=5, bidirectional=True)
        # 全连接
        self.fc = nn.Linear(5 * 2, 1)

        # # Attention
        # self.attention = AttentionSeq(5, hard=0.03)

    def attention_net(self, lstm_output, final_state):
        # batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size, -1, 1)  # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [batch, 64,5,4]
        x = x.view(x.size(0), -1)  # 将多维展为1维 [batch, 1280]
        x = x.unsqueeze(1)
        x = x.transpose(0, 1)
        # x = self.output(x)
        # x = x.transpose(0, 1)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        x = x.transpose(0, 1)
        # x = self.attention(x)
        x, attention = self.attention_net(x, final_hidden_state)
        output = self.fc(x)
        return output


# class AttentionSeq(nn.Module):
#     def __init__(self, hidden_dim, hard=0.0):
#         super(AttentionSeq, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.dense = nn.Linear(hidden_dim, hidden_dim)
#         self.hard = hard
#
#     def forward(self, features, mean=False):
#         # [batch,seq,dim]
#         batch_size, time_step, hidden_dim = features.size()
#         weight = nn.Tanh()(self.dense(features))
#
#         # mask给负无穷使得权重为0
#         mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
#         #       mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
#         mask_idx = mask_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)
#
#         # 注意这里torch.where意思是按照第一个参数的条件对每个元素进行检查，若满足条件，则使用第二个元素进行填充，若不满足，则使用第三个元素填充。
#         # 此时会填充一个极小的数----不能为零，具体请参考softmax中关于Tahn。
#         # torch.full_like是按照第一个参数的形状，填充第二个参数。
#         weight = torch.where(mask_idx == 1, weight,
#                              torch.full_like(mask_idx, (-2 ** 32 + 1)))
#         weight = weight.transpose(2, 1)
#
#         # 得出注意力分数
#         weight = torch.nn.Softmax(dim=2)(weight)
#         if self.hard != 0:  # hard mode
#             weight = torch.where(weight > self.hard, weight, torch.full_like(weight, 0))
#
#         if mean:
#             weight = weight.mean(dim=1)
#             weight = weight.unsqueeze(1)
#             weight = weight.repeat(1, hidden_dim, 1)
#         weight = weight.transpose(2, 1)
#         # 将注意力分数作用在输入值上
#         features_attention = weight * features
#         # 返回结果
#         return features_attention


def main():
    input = torch.ones((5, 6, 43, 39))  # 5个图片， 6个通道
    model = NN()
    output = model(input)
    print(output)
    test_input = torch.ones(1, )


if __name__ == '__main__':
    main()
