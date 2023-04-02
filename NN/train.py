"""模型搭建与训练"""
import os.path

from matplotlib import pyplot as plt

from model import *
from dataset import *
from datetime import datetime
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
import utils
from NN_CONST import *


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True


def train():
    total_training_loss_list = []  # 用于绘制损失趋势图
    total_test_loss_list = []

    min_loss = 999999
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
            if i + 1 == 28 / BATCH_SIZE:
                total_training_loss_list.append(total_training_loss)
            print(f"epoch:{epoch}  i:{i}   training_loss:{training_loss.item()}  "
                  f"total_training_loss:{total_training_loss}")
            training_loss.backward()
            optimizer.step()

        # 模型验证
        total_test_loss = 0.
        with torch.no_grad():
            for data in test_dataloader:
                test_inputs, test_labels = data
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                total_test_loss += test_loss.item()
                # 查找损失最小模型
                if total_test_loss < min_loss:
                    min_loss = total_test_loss
                    best_model = model
                total_test_loss_list.append(total_test_loss)
                print(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
                      f"total_testing_loss:{total_test_loss}")

    # 绘制损失图
    plt.plot(list(range(EPOCH)), total_training_loss_list, label="total training loss")
    plt.plot(list(range(EPOCH)), total_test_loss_list, label="total test loss")
    # plt.ylim([0, 0.002])
    plt.legend()
    os.makedirs(LOSS_PATH, exist_ok=True)
    plt.savefig(LOSS_PATH + rf"/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年损失(除{test_year}).png")
    plt.close()
    return best_model


if __name__ == '__main__':
    # 日志初始化
    logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s  %(message)s')
    # setup_seed(20)
    # 所有年份中选一年作为测试集，其他年份作为训练集，以不同的训练集循环训练多个模型
    for test_year in range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1):
        # 加载训练集
        train_dataset = TrainDataset(test_year)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
        # 加载测试集 用于查看损失
        test_dataset = TestDataset(test_year, test_year, train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
        # 加载模型
        model = LSTM_CNN(train_dataset.shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 开始训练
        logging.debug(f"开始训练1991-2019年模型({test_year}年除外)")
        print(f"开始训练1991-2019年模型({test_year}年除外)")
        start = datetime.now()
        best_model = train()
        end = datetime.now()
        logging.debug(f"模型训练完成,耗时:{end - start}")
        print("模型训练完成,耗时:", end - start)

        os.makedirs(MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
        model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-" \
                                  rf"{TRAIN_END_YEAR}年模型(除{test_year}年).pth"
        torch.save(best_model.state_dict(), model_path)
        print("保存模型文件:", model_path)

