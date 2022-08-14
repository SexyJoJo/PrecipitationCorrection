"""CNN + BiLSTM + Attention"""
import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,56,56]
            nn.Conv2d(  # 定义1个2维的卷积核
                in_channels=1,  # 输入通道的个数（一般灰度图片为1， 彩色RGB图片为3， 带透明度的RGBa图片为4）
                out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
                kernel_size=(5, 5),  # 卷积核的大小
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
                padding=2,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
            ),
            # 经过卷积层 输出[16,56,56] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过最大值池化 输出[16,28,28] 传入下一个卷积
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 输入个数与上层输出一致
                out_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2
            ),
            # 经过卷积 输出[32, 28, 28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,14,14] 传入输出层
        )

        # 第三层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # 输入个数与上层输出一致
                out_channels=64,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2
            ),
            # 经过卷积 输出[64, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
        )

        # 输出层（全连接）
        self.output = nn.Linear(in_features=32 * 7 * 7, out_features=1)

        # BiLSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, bidirectional=True)
        # 全连接
        self.fc = nn.Linear(5 * 2, 1)

        # Attention
        self.attention = AttentionSeq(5, hard=0.03)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        x = self.output(x)
        x = self.lstm(x)
        output = self.fc(x)  # 输出[50,10]
        return output


class AttentionSeq(nn.Module):

    def __init__(self, hidden_dim, hard=0):
        super(AttentionSeq, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.hard = hard

    def forward(self, features, mean=False):
        # [batch,seq,dim]
        batch_size, time_step, hidden_dim = features.size()
        weight = nn.Tanh()(self.dense(features))

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
        #       mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
        mask_idx = mask_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)

        # 注意这里torch.where意思是按照第一个参数的条件对每个元素进行检查，若满足条件，则使用第二个元素进行填充，若不满足，则使用第三个元素填充。
        # 此时会填充一个极小的数----不能为零，具体请参考softmax中关于Tahn。
        # torch.full_like是按照第一个参数的形状，填充第二个参数。
        weight = torch.where(mask_idx == 1, weight,
                             torch.full_like(mask_idx, (-2 ** 32 + 1)))
        weight = weight.transpose(2, 1)

        # 得出注意力分数
        weight = torch.nn.Softmax(dim=2)(weight)
        if self.hard != 0:  # hard mode
            weight = torch.where(weight > self.hard, weight, torch.full_like(weight, 0))

        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        # 将注意力分数作用在输入值上
        features_attention = weight * features
        # 返回结果
        return features_attention

