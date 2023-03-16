#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   RCM_correction
@file    :   p08trainandtest.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-1-29 19:19
@description:
模型训练和测试-ANN类型模型的训练
配合p07model.py中的ann_lstm混合模型，进行实验。
'''

# here put the import lib
import torch
import torch.utils.data
import numpy as np
from p07model import ANN_LSTM_Model
import tqdm
from visdom import Visdom
import random
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import utils
import utils_stat


# 固定随机种子
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)    # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(SEED)    # 为所有GPU设置随机种子

# 常量定义
MODEL_NAME = 'l1h64-lstm-d2'
INPUT_DIM = 8 + 2  # 模型输入维度
OUTPUT_DIM = 1     # 模型输出维度
NUM_LAYERS = 1     # 隐层数量
HIDDEN_DIM = 64    # 隐层维度
ACTIVATION = 'tanh'    # 激活函数
BATCHNORMFLAG = False  # BATCH NORM 标识
DROUPOUTFLAG = True    # DROPOUT 标识
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.001


# 生成标准化参数
def generate_orm_param():
    area = 'JSJ'
    case_num = 1
    # case_year = 1992
    case_date = '0131'
    case_time = '00'
    obs_month = 2

    data_all = []
    for case_year in range(1991, 2020):
        # 处理单个文件
        input_filename = rf'../data/202301-gridpoint/' \
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
    output_filename = rf'../data/202301-gridpoint/' \
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

        self.input_dim = INPUT_DIM  # 模型输入维度
        self.output_dim = OUTPUT_DIM     # 模型输出维度
        self.num_layers = NUM_LAYERS     # 隐层数量
        self.hidden_dim = HIDDEN_DIM    # 隐层维度
        self.activation = ACTIVATION    # 激活函数
        self.batchnormflag = BATCHNORMFLAG  # batch norm 标识
        self.droupoutflag = DROUPOUTFLAG    # dropout 标识

        self.batch_size = BATCH_SIZE
        self.__epochs = EPOCHS
        self.__lr = LR

    # 组织数据
    def organize_data(self):
        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []

        normpara_filename = rf'../data/202301-gridpoint/' \
                            rf'{self.area}_point_normparam_PRAVG_' \
                            rf'{self.case_date}{self.case_time}c{str(self.case_num).zfill(2)}_monthly.txt'
        normpara = np.loadtxt(normpara_filename, delimiter='\t', skiprows=1)

        for year in range(1991, 2020):
            input_filename = rf'../data/202301-gridpoint/' \
                             rf'{self.area}_point_PRAVG_{year}{self.case_date}' \
                             rf'{self.case_time}c{str(self.case_num).zfill(2)}' \
                             rf'_{year}_monthly.txt'
            data_one_file = np.loadtxt(input_filename, delimiter='\t', skiprows=1)
            # 归一化, zsore: x' = (x -avg)/std
            assert normpara.shape[1] == data_one_file.shape[1], '列数量不一致'
            assert data_one_file.shape[1] - 2 == self.input_dim + self.output_dim, '数据维度与模型结构不一致'
            for ix in range(2, normpara.shape[1]):
                data_one_file[:, ix] = (data_one_file[:, ix] - normpara[0, ix]) / normpara[1, ix]

            data_one_x = data_one_file[:, 2:data_one_file.shape[1]-1]
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
        model = ANN_LSTM_Model(input_dim=self.input_dim, output_dim=self.output_dim, num_layers=self.num_layers,
                         hidden_dim=self.hidden_dim, activation=self.activation,
                         batchnormflag=self.batchnormflag, droupoutflag=self.droupoutflag)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.__lr)

        best_model_loss = float('inf')
        # now = datetime.datetime.now()
        model_desp = str(model)
        MODEL_NAME = model_desp
        model_save = f'../result/model-{self.area}-c{str(self.case_num).zfill(2)}' \
                     f'-{self.case_date}-{self.case_time}' \
                     f'-y{self.test_year}' \
                     f'-{model_desp}.pth'
        print(model_save)
        # 训练损失和测试损失记录
        loss_list = []
        loss_save = f'../result/modelloss-{self.area}-c{str(self.case_num).zfill(2)}' \
                    f'-{self.case_date}-{self.case_time}-y{self.test_year}' \
                    f'-{model_desp}.txt'

        # visdom监控窗口
        viz = Visdom(env=f'{model_desp}')
        # 窗口初始化
        # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
        # viz.line([[0., 0.]], [0],
        #          win='tt_loss', update='append',
        #          opts=dict(title='train_test_loss', lenend=['trainloss', 'testloss']))

        # 模型训练
        for epoch in range(self.__epochs):
            model.train()
            training_loss = 0.
            loss_epoch = []
            # 设置进度条
            training_bar = tqdm.tqdm(enumerate(self.__train_loader), total=len(self.__train_loader),
                                     leave=False)
            for i, data in training_bar:
                x_train, y_train = data
                # x_train = x_train.view(len(x_train), 1, -1) # 增加一个len维度，以对应lstm
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                training_bar.set_description(f'epoch {epoch+1}/{self.__epochs} train')
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
                    # x_test = x_test.view(len(x_test), 1, -1)  # 增加一个len维度，以对应lstm
                    y_pred = model(x_test)
                    testing_loss += criterion(y_pred, y_test)
                    testing_bar.set_description(f'epoch {epoch+1}/{self.__epochs} test')
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


# 组织结果数据，恢复结果数据到经纬度格点上。
def organize_result_data(year=1991):
    # year = 1992
    area = 'JSJ'
    obs_month = 2
    begin_index = 2     # 列下标从2开始，包含经纬度；如果设置为4，则跳过经纬度
    model_name = MODEL_NAME
    # 1、提取模型
    model_file = f'../result/model-{area}-c01-0131-00-y{year}-{model_name}.pth'
    # model_file = f'../result/{model_name}/model-{area}-c01-0131-00-y{year}-{model_name}.pth'
    model = torch.load(model_file)
    # 2、提取数据，将格点数据生成为训练样本集合
    data_file = fr'../data/202301-gridpoint/{area}_point_PRAVG_{year}013100c01_{year}_monthly.txt'
    para_file = fr'../data/202301-gridpoint/{area}_point_normparam_PRAVG_013100c01_monthly.txt'

    data_oneyear = np.loadtxt(data_file, delimiter='\t', skiprows=1)
    para_norm = np.loadtxt(para_file, delimiter='\t', skiprows=1)
    assert data_oneyear.shape[1] == para_norm.shape[1], '列数量不一致'
    # 归一化
    for ix in range(begin_index, para_norm.shape[1]):
        data_oneyear[:, ix] = (data_oneyear[:, ix] - para_norm[0, ix]) / para_norm[1, ix]

    # 3、使用模型生成结果数据
    data_x = data_oneyear[:, begin_index : data_oneyear.shape[1]-1]
    data_x = torch.from_numpy(data_x).to(torch.float32)
    # data_x = data_x.view(len(data_x), 1, -1)  # 增加一个len维度，以对应lstm
    data_y = model(data_x)
    data_y = data_y.detach().numpy()
    # 反归一化
    data_y[:, 0] = data_y[:, 0] * para_norm[1, -1] + para_norm[0, -1]
    # print(data_y)

    # 4、将结果数据恢复成格点形状
    obs_ncfile = fr'../data/dividedobs/ChangJiang/{area}_obs_prec_rcm_{year}{str(obs_month).zfill(2)}.nc'
    pred_ncfile = fr'../result/{area}_PRcor_{year}013100c01_{year}_monthly-{model_name}.nc'

    obs_data = nc.Dataset(obs_ncfile)
    obs_lat = obs_data.variables["lat2d"][:]
    obs_lon = obs_data.variables["lon2d"][:]
    prec = obs_data.variables["prec"][:]
    pred_data = nc.Dataset(pred_ncfile, 'w', format='NETCDF4')
    pred_data.createDimension('x', obs_lat.shape[0])
    pred_data.createDimension('y', obs_lat.shape[1])
    pred_data.createVariable('lon2d', "f", ("x", "y"))
    pred_data.createVariable('lat2d', "f", ("x", "y"))
    pred_data.createVariable('prcor', "f", ("x", "y"))
    pred_data.variables["lon2d"][:] = obs_lon[:]
    pred_data.variables["lat2d"][:] = obs_lat[:]
    prcor = np.zeros(prec.shape, dtype=float)
    prcor_mask = np.ones(prec.shape, dtype=int)
    for ix in range(len(data_oneyear)):
        row = int(data_oneyear[ix, 0])
        col = int(data_oneyear[ix, 1])
        prcor[row, col] = data_y[ix, 0]
        prcor_mask[row, col] = 999
    mask = prcor_mask < 990
    prcor = np.ma.array(prcor, mask=mask)
    pred_data.variables["prcor"][:] = prcor[:]

    print(pred_ncfile)
    # 检查结果一致性
    for row in range(prec.shape[0]):
        for col in range(prec.shape[1]):
            if (isinstance(prec[row, col], np.ma.core.MaskedConstant) \
                    != isinstance(prcor[row, col], np.ma.core.MaskedConstant)):
                print(f'not match: {row}, {col}')

    obs_data.close()
    pred_data.close()


# 计算结果度量
def cal_measurement():
    # model_name = 'ann-l3h64-tanh-bn0-dp1'
    model_name = MODEL_NAME

    cases = []
    cors = []
    obses = []

    for year in range(1991, 2020):
        case_file = fr'../data/dividedcase/0131/CASE1/TIME00/ChangJiang/JSJ_PRAVG_{year}013100c01_{year}_monthly.nc'
        cor_file = fr'../result/JSJ_PRcor_{year}013100c01_{year}_monthly-{model_name}.nc'
        # cor_file = fr'../result/{model_name}-nc/JSJ_PRcor_{year}013100c01_{year}_monthly.nc'
        obs_file = fr'../data/dividedobs/ChangJiang/JSJ_obs_prec_rcm_{year}02.nc'

        case_data = nc.Dataset(case_file)
        cor_data = nc.Dataset(cor_file)
        obs_data = nc.Dataset(obs_file)

        case_pr = case_data.variables['PRAVG'][0, :] * 24 * 3600
        cor_pr = cor_data.variables['prcor'][:]
        obs_pr = obs_data.variables['prec'][:]

        assert cor_pr.shape == obs_pr.shape == case_pr.shape, '数据维度不一样'

        cor_pr.fill_value = np.nan
        cor_pr = cor_pr.filled()

        mask = obs_pr.mask
        obs_pr.fill_value = np.nan
        obs_pr = obs_pr.filled()

        # mask = (obs_pr == np.nan)
        case_pr = np.ma.array(case_pr, mask=mask)
        case_pr.fill_value = np.nan
        case_pr = case_pr.filled()

        cases.append(case_pr)
        cors.append(cor_pr)
        obses.append(obs_pr)

        # plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
        plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
        fig = plt.figure()
        fig.suptitle(f"{year}年2月")
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
        ax1.set_title("订正前")
        subfig = ax1.imshow(np.flip(case_pr, axis=0), norm=norm)

        ax2.set_title("订正后")
        ax2.imshow(np.flip(cor_pr, axis=0), norm=norm)

        ax3.set_title("obs")
        ax3.imshow(np.flip(obs_pr, axis=0), norm=norm)

        plt.colorbar(subfig, ax=[ax1, ax2, ax3], orientation="horizontal")

        plt.savefig(f"../result/{year}02")
        plt.close()

    print(len(cors))
    print(len(obses))
    print(cors[0])

    # TCC相关
    corr_tcc = utils.OtherUtils.cal_TCC(cors, obses)  # 订正后与真实值
    case_tcc = utils.OtherUtils.cal_TCC(cases, obses)  # 订正前与真实值
    tcc_img = utils.PaintUtils.paint_TCC(case_tcc, corr_tcc)
    tcc_img.savefig(rf"../result/TCC-JSJ_91-19年2月-{model_name}")
    print("tcc已保存")
    tcc_img.close()

    # ACC相关
    corr_acc = utils.OtherUtils.cal_ACC(cors, obses, False)
    case_acc = utils.OtherUtils.cal_ACC(cases, obses, False)
    acc_img = utils.PaintUtils.paint_ACC(range(1991, 2020), case_acc, corr_acc)
    acc_img.savefig(f"../result/ACC-JSJ_91-19年2月-{model_name}")
    print("tcc已保存")
    acc_img.close()


# 计算结果度量值
def cal_measure():
    model_name = MODEL_NAME

    cases = []
    cors = []
    obses = []

    case_obs_acc = []
    cor_obs_acc = []

    for year in range(1991, 2020):
        case_file = fr'../data/dividedcase/0131/CASE1/TIME00/ChangJiang/JSJ_PRAVG_{year}013100c01_{year}_monthly.nc'
        cor_file = fr'../result/{model_name}-nc/JSJ_PRcor_{year}013100c01_{year}_monthly-{model_name}.nc'
        # cor_file = fr'../result/{model_name}-nc/JSJ_PRcor_{year}013100c01_{year}_monthly.nc'
        obs_file = fr'../data/dividedobs/ChangJiang/JSJ_obs_prec_rcm_{year}02.nc'

        case_data = nc.Dataset(case_file)
        cor_data = nc.Dataset(cor_file)
        obs_data = nc.Dataset(obs_file)

        case_pr = case_data.variables['PRAVG'][0, :] * 24 * 3600
        cor_pr = cor_data.variables['prcor'][:]
        obs_pr = obs_data.variables['prec'][:] / 28

        assert cor_pr.shape == obs_pr.shape == case_pr.shape, '数据维度不一样'

        # cor_pr.fill_value = np.nan
        # cor_pr = cor_pr.filled()
        #
        # mask = obs_pr.mask
        # obs_pr.fill_value = np.nan
        # obs_pr = obs_pr.filled()
        #
        # # mask = (obs_pr == np.nan)
        # case_pr = np.ma.array(case_pr, mask=mask)
        # case_pr.fill_value = np.nan
        # case_pr = case_pr.filled()
        #
        # cases.append(case_pr)
        # cors.append(cor_pr)
        # obses.append(obs_pr)

        # 整理为一维数组
        case_line = []
        cor_line = []
        obs_line = []

        for row in range(case_pr.shape[0]):
            for col in range(case_pr.shape[1]):
                if (not isinstance(obs_pr[row, col], np.ma.core.MaskedConstant)):
                    obs_line.append(obs_pr[row, col])
                    case_line.append(case_pr[row, col])
                    cor_line.append(cor_pr[row,col])
        # 计算相关性
        acc_case_obs_one = utils_stat.correlation(case_line, obs_line)
        acc_cor_obs_one = utils_stat.correlation(cor_line, obs_line)
        case_obs_acc.append(acc_case_obs_one)
        cor_obs_acc.append(acc_cor_obs_one)

    print(case_obs_acc)
    print(cor_obs_acc)







if __name__ == '__main__':
    # generate_norm_param()

    # 模型训练
    ''' # 模型训练begin
    for year in range(1991, 2020):
        tt = AreaModelTrainTest(test_year=year)
        tt.organize_data()
        tt.train_test()
    # ''' # 模型训练end


    # organize_result_data()
    # ''' # 组织结果数据begin
    # 组织结果数据
    for i in range(1991, 2020):
        organize_result_data(year=i)
    # ''' # 组织结果数据end

    cal_measurement()

    # cal_measure()

    print('done')




