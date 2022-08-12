import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 为了每次的实验结果一致
# 设置超参数
epoches = 2
batch_size = 50
learning_rate = 0.001

# 训练集
train_data = torchvision.datasets.MNIST(
    root="./MINST/",  # 训练数据保存路径
    train=True,  # True为下载训练数据集，False为下载测试数据集
    transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
    download=True,  # 是否需要下载
)
# 显示训练集中的第一张图片
print(train_data.train_data.size())  # [60000,28,28]    60000张图片，图片为28*28像素
pic_matrix = train_data.train_data[0].numpy()
plt.imshow(pic_matrix, cmap='Greys')
plt.show()

# 测试集
test_data = torchvision.datasets.MNIST(root="./MINST/", train=False, download=True)
print(test_data.test_data.size())  # [10000, 28, 28]
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255
test_y = test_data.test_labels

# 将训练数据装入Loader中
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(  # 定义1个2维的卷积核
                in_channels=1,  # 输入通道的个数（一般灰度图片为1， 彩色RGB图片为3， 带透明度的RGBa图片为4）
                out_channels=16,  # 输出通道（卷积核）的个数（越多则能识别更多边缘特征，任务不复杂赋值16，复杂可以赋值64）
                kernel_size=(5, 5),  # 卷积核的大小
                stride=(1, 1),  # 卷积核在图上滑动，每隔一个扫描的次数
                padding=2,  # 周围填上多少圈的0, 一般为(kernel_size-1)/2
            ),
            # 经过卷积层 输出[16,28,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过最大值池化 输出[16,14,14] 传入下一个卷积
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
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
        )

        # 输出层（全连接）
        self.output = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)  # 输出[50,10]
        return output


def main():
    cnn = CNN()
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epoches):    # 一个epoch=一次前向计算与反向传播
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)  # batch_x=[50,1,28,28]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 为了实时显示准确率
            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

        test_output = cnn(test_x[:10])
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        print("预测值:", pred_y)
        print("真实值:", test_y[:10])


if __name__ == '__main__':
    main()
