#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   PrecipitationCorrection_qiaogit
@file    :   p01organizedata.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-1-16 19:19
@description:
组织一维数据
'''

# here put the import lib
import netCDF4 as nc
import numpy as np


# 组装格点展平的一维数据文件
def p01gridpointdataset():
    area = 'JSJ'
    case_num = 1
    case_year = 1992
    case_date = '0131'
    case_time = '00'
    obs_month = 2

    input_case_file = fr'../../divide area/divided case/{case_date}/CASE{str(case_num)}/TIME{case_time}/ChangJiang/' \
                      fr'{area}_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.nc'
    print(input_case_file)

    input_obs_file = fr'../../divide area/divided obs/ChangJiang/{area}_obs_prec_rcm_{case_year}{str(obs_month).zfill(2)}.nc'
    print(input_obs_file)

    nc_case_data = nc.Dataset(input_case_file)
    nc_obs_data = nc.Dataset(input_obs_file)
    print(nc_case_data)
    print(nc_obs_data)

    xlat_case = nc_case_data.variables['XLAT'][:]
    xlon_case = nc_case_data.variables['XLONG'][:]
    pravg_case = nc_case_data.variables['PRAVG'][:]
    print(xlat_case.shape, xlon_case.shape, pravg_case.shape)
    # print(pravg_case)
    print(np.max(pravg_case))
    print(np.min(pravg_case))

    xlat_obs = nc_obs_data.variables['lat2d'][:]
    xlon_obs = nc_obs_data.variables['lon2d'][:]
    pravg_obs = nc_obs_data.variables['prec'][:]
    print(xlat_obs.shape, xlon_obs.shape, pravg_obs.shape)
    print(np.max(pravg_obs))
    print(np.min(pravg_obs))

    assert pravg_case.shape[1:3] == pravg_obs.shape, "维度不一致"

    result = []
    for row in range(pravg_obs.shape[0]):
        for col in range(pravg_obs.shape[1]):
            if (not isinstance(pravg_obs[row, col], np.ma.core.MaskedConstant)):
                one_record = [row, col]
                one_record.append(xlat_obs[row, col])
                one_record.append(xlon_obs[row, col])
                one_record.extend(pravg_case[:, row, col] * 3600 * 24)
                one_record.append(pravg_obs[row, col])
                print(one_record)
                result.append(one_record)

    output_filename = rf'../data/202301-gridpoint/' \
                      rf'{area}_point_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.txt'

    fileheader = ['row', 'col', 'lat', 'lon']
    for i in range(pravg_case.shape[0]):
        fileheader.append(f'case_m{i + 1}')
    fileheader.append('obs')
    fileheader = '\t'.join(fileheader)
    np.savetxt(output_filename, result, header=fileheader, delimiter='\t', fmt="%.5f")
    # print(result)
    return result


def process_one_casefile(area='JSJ', case_num=1, case_year=1992,
                         case_date='0131', case_time='00', obs_month=2):
    input_case_file = fr'../../divide area/divided case/{case_date}/CASE{str(case_num)}/TIME{case_time}/ChangJiang/' \
                      fr'{area}_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.nc'
    print(input_case_file)

    input_obs_file = fr'../../divide area/divided obs/ChangJiang/{area}_obs_prec_rcm_{case_year}{str(obs_month).zfill(2)}.nc'
    print(input_obs_file)

    nc_case_data = nc.Dataset(input_case_file)
    nc_obs_data = nc.Dataset(input_obs_file)

    xlat_case = nc_case_data.variables['XLAT'][:]
    xlon_case = nc_case_data.variables['XLONG'][:]
    pravg_case = nc_case_data.variables['PRAVG'][:]

    xlat_obs = nc_obs_data.variables['lat2d'][:]
    xlon_obs = nc_obs_data.variables['lon2d'][:]
    pravg_obs = nc_obs_data.variables['prec'][:]

    assert pravg_case.shape[1:3] == pravg_obs.shape, "维度不一致"

    result = []
    for row in range(pravg_obs.shape[0]):
        for col in range(pravg_obs.shape[1]):
            if (not isinstance(pravg_obs[row, col], np.ma.core.MaskedConstant)):
                one_record = [row, col]
                one_record.append(xlat_obs[row, col])
                one_record.append(xlon_obs[row, col])
                one_record.extend(pravg_case[:, row, col] * 3600 * 24)
                one_record.append(pravg_obs[row, col])
                # print(one_record)
                result.append(one_record)
    return result, pravg_case.shape[0]


def porcess_all_file():
    area = 'JSJ'
    case_num = 1
    # case_year = 1992
    case_date = '0131'
    case_time = '00'
    obs_month = 2

    for case_year in range(1991, 2020):
    # for case_year in range(1994, 1995):
        # 处理单个文件
        result, monthnum = process_one_casefile(area, case_num, case_year, case_date, case_time, obs_month)

        # 写入文件
        output_filename = rf'../../divide area/data/202301-gridpoint/' \
                          rf'{area}_point_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.txt'

        fileheader = ['row', 'col', 'lat', 'lon']
        for i in range(monthnum):
            fileheader.append(f'case_m{i + 1}')
        fileheader.append('obs')
        fileheader = '\t'.join(fileheader)
        np.savetxt(output_filename, result, header=fileheader, delimiter='\t', fmt="%.5f")
        # print(result)


def z01():
    porcess_all_file()


# 查看1994年数据有什么问题
def z02():
    # input_case_file = fr'../data/dividedcase/{case_date}/CASE{str(case_num)}/TIME{case_time}/ChangJiang/' \
    #                  fr'{area}_PRAVG_{case_year}{case_date}{case_time}c{str(case_num).zfill(2)}_{case_year}_monthly.nc'
    input_case_file = r'../data/dividedcase/0131/CASE1/TIME00/ChangJiang/JSJ_PRAVG_1994013100c01_1994_monthly.nc'
    print(input_case_file)

    input_obs_file = fr'../data/dividedobs/ChangJiang/JSJ_obs_prec_rcm_199402.nc'
    print(input_obs_file)

    nc_case_data = nc.Dataset(input_case_file)
    nc_obs_data = nc.Dataset(input_obs_file)

    xlat_case = nc_case_data.variables['XLAT'][:]
    xlon_case = nc_case_data.variables['XLONG'][:]
    pravg_case = nc_case_data.variables['PRAVG'][:]

    xlat_obs = nc_obs_data.variables['lat2d'][:]
    xlon_obs = nc_obs_data.variables['lon2d'][:]
    pravg_obs = nc_obs_data.variables['prec'][:]

    assert pravg_case.shape[1:3] == pravg_obs.shape, "维度不一致"
    print(pravg_case.shape)
    print(pravg_obs.shape)
    print(pravg_case[0, 27, 15] * 3600 * 24)

    print(pravg_obs[27, 14])

    print(isinstance(pravg_obs[27, 14], np.ma.core.MaskedConstant))


def z03():
    input_obs_file1994 = fr'../data/dividedobs/ChangJiang/JSJ_obs_prec_rcm_199402.nc'
    input_obs_file1993 = fr'../data/dividedobs/ChangJiang/JSJ_obs_prec_rcm_199302.nc'

    obs_data1994 = nc.Dataset(input_obs_file1994)
    obs_data1993 = nc.Dataset(input_obs_file1993)

    obs_pr1994 = obs_data1994.variables['prec'][:]
    obs_pr1993 = obs_data1993.variables['prec'][:]

    mask1994 = obs_pr1994.mask
    mask1993 = obs_pr1993.mask

    print(mask1994)
    print(mask1993)

    for row in range(mask1993.shape[0]):
        for col in range(mask1993.shape[1]):
            if (mask1993[row, col] != mask1994[row, col]):
                print(row, col)


if __name__ == '__main__':
    # p01gridpointdataset()
    # porcess_all_file()
    z01()
    # z02()
    # z03()
    print('done')
