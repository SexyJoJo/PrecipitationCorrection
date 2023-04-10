#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   RCM_correction
@file    :   p05organizeresultdata.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-1-24 10:10
@description:
组织结果数据，数据组织恢复到经纬度网格上
1、提取模型
2、提取数据，将格点数据生成为训练样本集合
3、使用模型生成结果数据
4、将结果数据恢复成格点形状
'''

# here put the import lib
import os

import netCDF4 as nc
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import utils


# 组织结果数据，恢复结果数据到经纬度格点上。
def organize_result_data(year=1991):
    # year = 1992
    area = 'JSJ'
    obs_month = 2
    begin_index = 4  # 列下标从2开始，包含经纬度；如果设置为4，则跳过经纬度
    model_name = 'ann-l1h4-tanh-bn0-dp1除经纬度'
    # 1、提取模型
    model_file = f'../result/{model_name}/model-{area}-c01-0131-00-y{year}-{model_name}.pth'
    # model_file = f'../result/{model_name}/model-{area}-c01-0131-00-y{year}-{model_name}.pth'
    model = torch.load(model_file)
    # 2、提取数据，将格点数据生成为训练样本集合
    data_file = fr'../../divide area/data/202301-gridpoint/{area}_point_PRAVG_{year}013100c01_{year}_monthly.txt'
    para_file = fr'../../divide area/data/202301-gridpoint/{area}_point_normparam_PRAVG_013100c01_monthly.txt'

    data_oneyear = np.loadtxt(data_file, delimiter='\t', skiprows=1)
    para_norm = np.loadtxt(para_file, delimiter='\t', skiprows=1)
    assert data_oneyear.shape[1] == para_norm.shape[1], '列数量不一致'
    # 归一化
    for ix in range(begin_index, para_norm.shape[1]):
        data_oneyear[:, ix] = (data_oneyear[:, ix] - para_norm[0, ix]) / para_norm[1, ix]

    # 3、使用模型生成结果数据
    data_x = data_oneyear[:, begin_index: data_oneyear.shape[1] - 1]
    data_x = torch.from_numpy(data_x).to(torch.float32)
    # data_x = data_x.view(len(data_x), 1, -1)  # 增加一个len维度，以对应lstm
    data_y = model(data_x)
    data_y = data_y.detach().numpy()
    # 反归一化
    data_y[:, 0] = data_y[:, 0] * para_norm[1, -1] + para_norm[0, -1]
    # print(data_y)

    # 4、将结果数据恢复成格点形状
    obs_ncfile = fr'../../divide area/divided obs/ChangJiang/{area}_obs_prec_rcm_{year}{str(obs_month).zfill(2)}.nc'
    os.makedirs(fr'../result/{model_name}-nc', exist_ok=True)
    pred_ncfile = fr'../result/{model_name}-nc/{area}_PRcor_{year}013100c01_{year}_monthly-{model_name}.nc'

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
    print(utils.OtherUtils.mse(prcor, prec))
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
    # model_name = 'lstm-l2h64d2'
    # model_name = 'ann-l1h32-tanh-bn0-dp1'
    model_name = 'ann-l1h4-tanh-bn0-dp1'


    cases = []
    cors = []
    obses = []

    for year in range(1991, 2020):
        case_file = fr'../../divide area/divided case/0131/CASE1/TIME00/ChangJiang/JSJ_PRAVG_{year}013100c01_{year}_monthly.nc'
        cor_file = fr'../result/{model_name}-nc/JSJ_PRcor_{year}013100c01_{year}_monthly-{model_name}.nc'
        # cor_file = fr'../result/{model_name}-nc/JSJ_PRcor_{year}013100c01_{year}_monthly.nc'
        obs_file = fr'../../divide area/divided obs/ChangJiang/JSJ_obs_prec_rcm_{year}02.nc'

        case_data = nc.Dataset(case_file)
        cor_data = nc.Dataset(cor_file)
        obs_data = nc.Dataset(obs_file)

        case_pr = case_data.variables['PRAVG'][0, :] * 24 * 3600
        cor_pr = cor_data.variables['prcor'][:]
        obs_pr = obs_data.variables['prec'][:] / 28

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

        os.makedirs(fr'../result/{model_name}-result', exist_ok=True)
        plt.savefig(f"../result/{model_name}-result/{year}02")
        plt.close()

    print(len(cors))
    print(len(obses))
    print(cors[0])

    # TCC相关
    corr_tcc = utils.OtherUtils.cal_TCC(cors, obses)  # 订正后与真实值
    case_tcc = utils.OtherUtils.cal_TCC(cases, obses)  # 订正前与真实值
    tcc_img = utils.PaintUtils.paint_TCC(case_tcc, corr_tcc)
    tcc_img.savefig(rf"../result/{model_name}-result/TCC-JSJ_91-19年2月-{model_name}")
    print("tcc已保存")
    tcc_img.close()

    # ACC相关
    corr_acc = utils.OtherUtils.cal_ACC(cors, obses)
    case_acc = utils.OtherUtils.cal_ACC(cases, obses)
    acc_img = utils.PaintUtils.paint_ACC(range(1991, 2020), case_acc, corr_acc)
    acc_img.savefig(f"../result/{model_name}-result/ACC-JSJ_91-19年2月-{model_name}")
    print("acc已保存")
    print(rf"../result/{model_name}-result/ACC-JSJ_91-19年2月-{model_name}")
    acc_img.close()


if __name__ == '__main__':
    for i in range(1991, 2020):
        organize_result_data(year=i)

    cal_measurement()

    # organize_result_data(year=1994)

    print('done')
