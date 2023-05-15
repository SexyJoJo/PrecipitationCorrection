import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM_CNN(nn.Module):
    def __init__(self, shape):
        super(LSTM_CNN, self).__init__()
        self.shape = shape

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

            # nn.Conv2d(
            #     in_channels=16,  # 输入个数与上层输出一致
            #     out_channels=32,
            #     kernel_size=(3, 3),
            #     stride=(1, 1),
            #     padding=1
            # ),
            # nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 输入个数与上层输出一致
                out_channels=1,
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


class LSTM(nn.Module):
    def __init__(self, shape):
        super(LSTM, self).__init__()
        self.shape = shape
        input_size = shape[2] * shape[3]
        hidden_size = shape[2] * shape[3]
        num_layers = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, shape[2] * shape[3])

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1], -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(shape[0], 1, shape[2], shape[3])
        return x


class LSTM11(nn.Module):
    def __init__(self, shape):
        super(LSTM11, self).__init__()
        self.shape = shape
        input_size = 1
        hidden_size = 64
        num_layers = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shape = self.shape
        x = x.view(len(x), 8, -1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc(x)
        # x = x.view(shape[0], 1, shape[2], shape[3])
        return x


class ANN(nn.Module):
    def __init__(self, shape):
        super(ANN, self).__init__()

        net = nn.Sequential()
        net.add_module(r'inputlayer',
                       nn.Linear(in_features=shape[1], out_features=4, bias=True))
        net.add_module(r'inputact', nn.Tanh())
        net.add_module(rf'hidlayer-1',
                       nn.Linear(in_features=4, out_features=4, bias=True))
        net.add_module(rf'drop-1', nn.Dropout())
        net.add_module(rf'hidact-1', nn.Tanh())
        net.add_module(r'outputlayer',
                       nn.Linear(in_features=4, out_features=1, bias=True))
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class ANN33(nn.Module):
    def __init__(self, shape):
        super(ANN33, self).__init__()

        net = nn.Sequential()
        net.add_module(r'inputlayer',
                       nn.Linear(in_features=9 * shape[1], out_features=128, bias=True))
        net.add_module(r'inputact', nn.Tanh())
        net.add_module(rf'hidlayer-1',
                       nn.Linear(in_features=128, out_features=64, bias=True))
        # net.add_module(rf'drop-1', nn.Dropout())
        net.add_module(rf'hidact-1', nn.Tanh())
        net.add_module(r'outputlayer',
                       nn.Linear(in_features=64, out_features=1, bias=True))
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class UNet(nn.Module):
    def __init__(self, shape):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.shape = shape

        # # 3层Unet
        # self.encoder1 = conv_block(8, 16)
        # self.pool1 = nn.MaxPool2d(2)
        #
        # self.encoder2 = conv_block(16, 32)
        # self.pool2 = nn.MaxPool2d(2)
        #
        # self.encoder3 = conv_block(32, 64)
        # self.pool3 = nn.MaxPool2d(2)
        #
        # self.middle = conv_block(64, 128)
        #
        # self.upconv1 = upconv_block(128, 64)
        # self.decoder1 = conv_block(128, 64)
        #
        # self.upconv2 = upconv_block(64, 32)
        # self.decoder2 = conv_block(64, 32)
        #
        # self.upconv3 = upconv_block(32, 16)
        # self.decoder3 = conv_block(32, 16)
        #
        # self.output = nn.Conv2d(16, 1, kernel_size=1)

        # 2层Unet
        self.encoder1 = conv_block(8, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = conv_block(32, 64)

        self.upconv1 = upconv_block(64, 32)
        self.decoder1 = conv_block(64, 32)

        self.upconv2 = upconv_block(32, 16)
        self.decoder2 = conv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # # 3层Unet
        # enc1 = self.encoder1(x)
        # x = self.pool1(enc1)
        #
        # enc2 = self.encoder2(x)
        # x = self.pool2(enc2)
        #
        # enc3 = self.encoder3(x)
        # x = self.pool3(enc3)
        #
        # x = self.middle(x)
        #
        # x = self.upconv1(x)
        # x = torch.cat([x, nn.Upsample(size=(x.size(2), x.size(3)), mode='nearest')(enc3)], dim=1)
        # x = self.decoder1(x)
        #
        # x = self.upconv2(x)
        # x = torch.cat([x, nn.Upsample(size=(x.size(2), x.size(3)), mode='nearest')(enc2)], dim=1)
        # x = self.decoder2(x)
        #
        # x = self.upconv3(x)
        # x = torch.cat([x, nn.Upsample(size=(x.size(2), x.size(3)), mode='nearest')(enc1)], dim=1)
        # x = self.decoder3(x)
        #
        # x = self.output(x)
        # x = nn.functional.interpolate(x, size=(self.shape[2], self.shape[3]), mode='bilinear', align_corners=False)

        # 3层Unet
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)

        enc2 = self.encoder2(x)
        x = self.pool2(enc2)

        x = self.middle(x)

        x = self.upconv1(x)
        x = torch.cat([x, nn.Upsample(size=(x.size(2), x.size(3)), mode='nearest')(enc2)], dim=1)
        x = self.decoder1(x)

        x = self.upconv2(x)
        x = torch.cat([x, nn.Upsample(size=(x.size(2), x.size(3)), mode='nearest')(enc1)], dim=1)
        x = self.decoder2(x)

        x = self.output(x)
        x = nn.functional.interpolate(x, size=(self.shape[2], self.shape[3]), mode='bilinear', align_corners=False)
        return x
