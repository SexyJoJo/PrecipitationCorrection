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
from NN_CONST import *
from visdom import Visdom


torch.manual_seed(42)


def train():
    viz = Visdom(env='PR')
    # 窗口初始化
    # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    # viz.line([[0., 0.]], [0],
    #          win='tt_loss', update='append',
    #          opts=dict(title='train_test_loss', legend=['trainloss', 'testloss']))

    total_training_loss_list = []  # 用于绘制损失趋势图
    total_test_loss_list = []

    best_model_loss = float('inf')
    for epoch in range(EPOCH):
        # 模型训练
        model.train()
        training_loss = 0.
        loss_epoch = []
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            # inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            print(f"epoch:{epoch}  i:{i}   loss:{loss.item()}  "
                  f"total_training_loss:{training_loss}")
        training_one_loss = training_loss / len(train_dataloader)
        loss_epoch.append(training_one_loss)

        # 模型验证
        model.eval()
        testing_loss = 0.
        with torch.no_grad():
            for data in test_dataloader:
                test_inputs, test_labels = data
                # test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                testing_loss += test_loss

        testing_loss = testing_loss.item() / len(test_dataloader)
        loss_epoch.append(testing_loss)

        # 查找损失最小模型
        os.makedirs(MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
        model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-" \
                                  rf"{TRAIN_END_YEAR}年模型(除{test_year}年).pth"
        if testing_loss < best_model_loss and epoch > 20:
            best_model_loss = testing_loss
            torch.save(model, model_path, _use_new_zipfile_serialization=False)

        viz.line([[training_one_loss, testing_loss]], [epoch],
                 win=f'tt_loss_{test_year}', update='append',
                 opts=dict(title=f'train_test_loss_{test_year}', legend=['trainloss', 'testloss']))
        # print(f"epoch:{epoch}   testing_loss:{test_loss.item()}  "
        #       f"total_testing_loss:{testing_loss}")
        # for name, param in model.state_dict().items():
        #     print(f"{name}: {param}")

        total_training_loss_list.append(training_loss)
        total_test_loss_list.append(testing_loss)

    # 绘制损失图
    plt.plot(list(range(EPOCH)), total_training_loss_list, label="total training loss")
    plt.plot(list(range(EPOCH)), total_test_loss_list, label="total test loss")
    # plt.ylim([0, 0.002])
    plt.legend()
    os.makedirs(LOSS_PATH, exist_ok=True)
    plt.savefig(LOSS_PATH + rf"/{AREA}_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}年损失(除{test_year}).png")
    plt.close()
    # return model


if __name__ == '__main__':
    # 日志初始化
    logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s  %(message)s')
    # setup_seed(20)
    # 所有年份中选一年作为测试集，其他年份作为训练集，以不同的训练集循环训练多个模型
    for test_year in range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1):
        # 加载训练集
        train_dataset = TrainDataset(test_year)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 加载测试集 用于查看损失
        test_dataset = TestDataset(test_year, test_year, train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # 加载模型
        if DATA_FORMAT == 'map':
            model = LSTM_CNN(train_dataset.shape)
        elif DATA_FORMAT == 'gird11':
            model = ANN(train_dataset.shape)
        elif DATA_FORMAT == 'grid33':
            model = ANN33(train_dataset.shape)
        else:
            print('请选择正确的模型')
            break
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # 开始训练
        logging.debug(f"开始训练1991-2019年模型({test_year}年除外)")
        print(f"开始训练1991-2019年模型({test_year}年除外)")
        start = datetime.now()
        train()
        end = datetime.now()
        logging.debug(f"模型训练完成,耗时:{end - start}")
        print("模型训练完成,耗时:", end - start)
