"""模型搭建与训练"""
import os.path
from model import *
from dataset import *
from datetime import datetime
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from NN_CONST import *
from visdom import Visdom
from torch.optim.lr_scheduler import StepLR


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # 查找损失最小模型
        os.makedirs(MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
        best_model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-" \
                                       rf"{TRAIN_END_YEAR}年模型(除{test_year}年).pth"
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, best_model_path)  # save checkpoint
        self.val_loss_min = val_loss


def train():
    viz = Visdom(env='PR')
    early_stopping = EarlyStopping(patience=7, verbose=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
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
            # print(f"epoch:{epoch}  i:{i}   loss:{loss.item()}  "
            #       f"total_training_loss:{training_loss}")
        training_one_loss = training_loss / len(train_dataloader)
        loss_epoch.append(training_one_loss)

        # 模型验证
        model.eval()
        testing_loss = 0.
        with torch.no_grad():
            for data in valid_dataloader:
                valid_inputs, valid_labels = data
                # test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                valid_outputs = model(valid_inputs)
                valid_loss = criterion(valid_outputs, valid_labels)
                testing_loss += valid_loss

        testing_loss = testing_loss.item() / len(test_dataloader)
        loss_epoch.append(testing_loss)

        # visdom中绘制损失图
        viz.line([[training_one_loss, testing_loss]], [epoch],
                 win=f'tt_loss_{test_year}', update='append',
                 opts=dict(title=f'train_test_loss_{test_year}', legend=['trainloss', 'testloss']))

        # 早停
        if EARLY_STOP:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            # 查找损失最小模型
            os.makedirs(MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}", exist_ok=True)
            best_model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-" \
                                           rf"{TRAIN_END_YEAR}年模型(除{test_year}年).pth"
            if testing_loss < best_model_loss and epoch > 20:
                best_model_loss = testing_loss
                torch.save(model, best_model_path, _use_new_zipfile_serialization=False)

        # 动态学习率
        if STEP_LR:
            scheduler.step()

        # print(f"epoch:{epoch}   testing_loss:{valid_loss.item()}  "
        #       f"total_testing_loss:{testing_loss}")
        # for name, param in model.state_dict().items():
        #     print(f"{name}: {param}")

    # 保存最终模型
    # final_model_path = MODEL_PATH + rf"/{DATE}/{CASE_NUM}/{TIME}/{BASIN}/{AREA}_{TRAIN_START_YEAR}-" \
    #                                 rf"{TRAIN_END_YEAR}年模型(除{test_year}年).pth"
    # torch.save(model, final_model_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    # 设定种子
    torch.manual_seed(42)
    # 日志初始化
    logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s  %(message)s')
    # 所有年份中选一年作为测试集，一年作为验证集，其他年份作为训练集，以不同的训练集循环训练多个模型
    for test_year in range(TRAIN_START_YEAR + 1, TRAIN_END_YEAR + 1):
        valid_year = test_year - 1
        # 加载训练集
        train_dataset = TrainDataset([valid_year, test_year])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 加载验证集
        valid_dataset = TestDataset(valid_year, valid_year, train_dataset)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # 加载测试集
        test_dataset = TestDataset(test_year, test_year, train_dataset)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # 加载模型
        if DATA_FORMAT == 'map' and MODEL == 'LSTM_CNN':
            model = LSTM_CNN(train_dataset.shape)
        elif DATA_FORMAT == 'map' and MODEL == 'LSTM':
            model = LSTM(train_dataset.shape)
        elif DATA_FORMAT == 'grid11' and MODEL == 'LSTM11':
            model = LSTM11(train_dataset.shape)
        elif DATA_FORMAT == 'grid11' and MODEL == 'ANN':
            model = ANN(train_dataset.shape)
        elif DATA_FORMAT == 'grid11' and MODEL == 'ANN_h16':
            model = ANN_h16(train_dataset.shape)
        elif DATA_FORMAT == 'grid33' and MODEL == 'ANN33':
            model = ANN33(train_dataset.shape)
        elif DATA_FORMAT == 'map' and MODEL == 'UNet':
            model = UNet(train_dataset.shape)
        else:
            print('请选择正确的模型与数据格式')
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
