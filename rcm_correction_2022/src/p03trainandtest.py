#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   RCM_correction
@file    :   p03trainandtest.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-1-19 19:19
@description:
模型训练和测试-LSTM类型模型的训练
'''

# here put the import lib
import torch
import torch.utils.data
import numpy as np
from p02model import LSTMModel
import tqdm
from visdom import Visdom
import datetime

# 生成标准化参数
def generate_norm_param():
    area = 'JSJ'
    case_num = 1
    # case_year = 1992
    case_date = '0131'
    case_time = '00'
    obs_month = 2

    data_all = []
    for case_year in range(1991, 2020):
        # 处理单个文件
        input_filename = rf'../../divide area/data/202301-gridpoint/' \
                         rf'{area}_point_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.txt'
        data_one_file = np.loadtxt(input_filename, delimiter='\t', skiprows=1)
        # print(data_one_file.shape, data_all.shape)
        # if not data_all:
        #     assert data_all.shape[1] == data_one_file.shape[1], '列数量不一致'
        if len(data_all) == 0:
            data_all = data_one_file
        else:
            assert data_all.shape[1] == data_one_file.shape[1], '列数量不一致'
            data_all = np.concatenate([data_all, data_one_file], axis=0)
            print(data_all.shape)

    # scaler = StandardScaler()
    # data = scaler.fit_transform(data_all)
    # print(scaler.mean_, np.sqrt(scaler.var_))

    mean = np.mean(data_all, axis=0)
    std = np.std(data_all, axis=0)
    print(mean, std)
    result = np.concatenate([mean, std], axis=0)
    result = result.reshape((2, -1))
    print(result)
    # print(result.shape)

    # 写入文件
    output_filename = rf'../../divide area/data/202301-gridpoint/' \
                      rf'{area}_point_normparam_PRAVG_{case_date}{case_time}c{str(case_num).zfill(2)}_monthly.txt'

    fileheader = ['row', 'col', 'lat', 'lon']
    for i in range(result.shape[1] - 5):
        fileheader.append(f'case_m{i + 1}')
    fileheader.append('obs')
    fileheader = '\t'.join(fileheader)
    np.savetxt(output_filename, result, header=fileheader, delimiter='\t', fmt="%.5f")


# 模型训练类，针对LSTM的数据
class AreaModelTrainTest():
    def __init__(self, area='JSJ', case_num=1,
                 case_date='0131', case_time='00',
                 obs_month=2, test_year=2019):
        self.area = area
        self.case_num = case_num
        self.test_year = test_year  #按照年份进行测试，该年度做为测试样本，其他年度作为训练样本
        self.case_date = case_date
        self.case_time = case_time
        self.obs_month = obs_month
        self.__train_dataset = None
        self.__test_dataset = None
        self.__train_loader = None
        self.__test_loader = None
        self.batch_size = 128
        self.input_dim = 8
        self.output_dim = 1
        self.num_layers = 1
        self.hidden_dim = 64
        self.epochs = 50


    # 组织数据
    def organize_data(self):
        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []

        normpara_filename = rf'../../divide area/data/202301-gridpoint/' \
                            rf'{self.area}_point_normparam_PRAVG_' \
                            rf'{self.case_date}{self.case_time}c{str(self.case_num).zfill(2)}_monthly.txt'
        normpara = np.loadtxt(normpara_filename, delimiter='\t', skiprows=1)

        for year in range(1991, 2020):
            input_filename = rf'../../divide area/data/202301-gridpoint/' \
                             rf'{self.area}_point_PRAVG_{year}{self.case_date}' \
                             rf'{self.case_time}c{str(self.case_num).zfill(2)}' \
                             rf'_{year}_monthly.txt'
            data_one_file = np.loadtxt(input_filename, delimiter='\t', skiprows=1)
            # 归一化, zsore: x' = (x -avg)/std
            assert normpara.shape[1] == data_one_file.shape[1], '列数量不一致'
            assert data_one_file.shape[1] - 4 == self.input_dim + self.output_dim, '数据维度与模型结构不一致'
            for ix in range(4, normpara.shape[1]):
                data_one_file[:, ix] = (data_one_file[:, ix] - normpara[0, ix]) / normpara[1, ix]

            data_one_x = data_one_file[:, 4:data_one_file.shape[1]-1]
            data_one_y = data_one_file[:, -1]
            if year == self.test_year:
                test_data_x = data_one_x
                test_data_y = data_one_y
            else:
                if len(train_data_x) == 0:
                    train_data_x = data_one_x
                    train_data_y = data_one_y
                else:
                    train_data_x = np.concatenate([train_data_x, data_one_x], axis=0)
                    train_data_y = np.concatenate([train_data_y, data_one_y], axis=0)
        if len(train_data_y.shape) == 1:
            train_data_y = train_data_y.reshape(-1, 1)
        if len(test_data_y.shape) == 1:
            test_data_y = test_data_y.reshape(-1, 1)
        train_data_x_tensor = torch.from_numpy(train_data_x).to(torch.float32)
        train_data_y_tensor = torch.from_numpy(train_data_y).to(torch.float32)
        test_data_x_tensor = torch.from_numpy(test_data_x).to(torch.float32)
        test_data_y_tensor = torch.from_numpy(test_data_y).to(torch.float32)
        self.__train_dataset = torch.utils.data.TensorDataset(train_data_x_tensor, train_data_y_tensor)
        self.__test_dataset = torch.utils.data.TensorDataset(test_data_x_tensor, test_data_y_tensor)
        self.__train_loader = torch.utils.data.DataLoader(self.__train_dataset,
                                                          batch_size=self.batch_size, shuffle=True)
        self.__test_loader = torch.utils.data.DataLoader(self.__test_dataset,
                                                         batch_size=self.batch_size, shuffle=False)


    # 训练和测试过程
    def train_test(self):
        model = LSTMModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                          num_layers=self.num_layers, output_dim=self.output_dim,
                          bidirect=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_model_loss = float('inf')
        # now = datetime.datetime.now()
        model_desp = str(model)
        model_save = f'../result/model-{self.area}-c{str(self.case_num).zfill(2)}' \
                     f'-{self.case_date}-{self.case_time}' \
                     f'-y{self.test_year}-l{self.num_layers}h{self.hidden_dim}' \
                     f'-{model_desp}.pth'
        print(model_save)
        # 训练损失和测试损失记录
        loss_list = []
        loss_save = f'../result/modelloss-{self.area}-c{str(self.case_num).zfill(2)}' \
                    f'-{self.case_date}-{self.case_time}' \
                    f'-y{self.test_year}-l{self.num_layers}h{self.hidden_dim}' \
                    f'-{model_desp}.txt'

        # visdom监控窗口
        viz = Visdom()
        # 窗口初始化
        # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
        # viz.line([[0., 0.]], [0],
        #          win='tt_loss', update='append',
        #          opts=dict(title='train_test_loss', lenend=['trainloss', 'testloss']))

        # 模型训练
        for epoch in range(self.epochs):
            model.train()
            training_loss = 0.
            loss_epoch = []
            # 设置进度条
            training_bar = tqdm.tqdm(enumerate(self.__train_loader), total=len(self.__train_loader),
                                     leave=False)
            for i, data in training_bar:
                x_train, y_train = data
                x_train = x_train.view(len(x_train), 1, -1) # 增加一个len维度，以对应lstm
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                training_bar.set_description(f'epoch {epoch}/{self.epochs} train')
                training_bar.set_postfix(training_loss=format(training_loss, '.5f'))

            training_one_loss = training_loss / len(self.__train_loader)
            loss_epoch.append(training_one_loss)


            # 模型验证
            model.eval()
            testing_loss = 0.
            with torch.no_grad():
                testing_bar = tqdm.tqdm(self.__test_loader, leave=False, colour='white')
                for data in testing_bar:
                    x_test, y_test = data
                    x_test = x_test.view(len(x_test), 1, -1)  # 增加一个len维度，以对应lstm
                    y_pred = model(x_test)
                    testing_loss += criterion(y_pred, y_test)
                    testing_bar.set_description(f'epoch {epoch}/{self.epochs} test')
                    testing_bar.set_postfix(testing_loss=format(testing_loss, '.5f'))

            testing_loss = testing_loss.item() / len(self.__test_loader)
            loss_epoch.append(testing_loss)
            loss_list.append(loss_epoch)
            if testing_loss < best_model_loss:
                best_model_loss = testing_loss
                torch.save(model, model_save, _use_new_zipfile_serialization=False)

            viz.line([[training_one_loss, testing_loss]], [epoch],
                     win=f'tt_loss_{self.test_year}', update='append',
                     opts=dict(title=f'train_test_loss_{self.test_year}', lenend=['trainloss', 'testloss']))

        file_head = 'training_loss\ttesting_loss'
        np.savetxt(loss_save, loss_list, header=file_head, delimiter='\t', fmt="%.8f")


if __name__ == '__main__':
    # generate_norm_param()
    for year in range(1991, 2020):
        tt = AreaModelTrainTest(test_year=year)
        tt.organize_data()
        tt.train_test()

    print('done')




