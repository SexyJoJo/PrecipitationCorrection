import os

import numpy as np
from torch.utils.data import Dataset
from NN_CONST import *
import torch
import utils

CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)


class TrainDataset(Dataset):
    def __init__(self, JUMP_YEAR):
        super().__init__()
        self.invalid_girds, self.valid_grids = utils.ObsParser.get_na_index(OBS_DIR, AREA)

        self.case_data = utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR)
        self.shape = self.case_data.shape
        self.months = utils.OtherUtils.get_predict_months(DATE, self.shape[1])
        self.obs_data = utils.ObsParser.get_many_2d_pravg(OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, self.months, JUMP_YEAR)
        all_case_data = utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA)
        all_obs_data = utils.ObsParser.get_many_2d_pravg(OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, self.months)
        # 去除异常值
        for i, j in self.invalid_girds:
            self.case_data[:, :, i, j] = np.nan
            self.obs_data[:, :, i, j] = np.nan
            all_case_data[:, :, i, j] = np.nan
            all_obs_data[:, :, i, j] = np.nan
        self.case_grids_means = np.mean(all_case_data, axis=0)
        self.obs_grids_means = np.mean(all_obs_data, axis=0)
        if DATA_FORMAT == 'map':
            self.case_data = np.nan_to_num(self.case_data, nan=0.0)
            self.obs_data = np.nan_to_num(self.obs_data, nan=0.0)
            self.obs_data = self.obs_data[:, 0: 1, :, :]

        # 归一化
        self.case_means, self.case_stds = utils.OtherUtils.cal_mean_std(all_case_data)
        self.obs_means, self.obs_stds = utils.OtherUtils.cal_mean_std(all_obs_data)
        self.min = np.min(self.case_data)
        self.max = np.max(self.case_data)
        if NORMALIZATION == 'zscore':
            self.case_data = utils.OtherUtils.zscore_normalization(self.case_data, self.case_means, self.case_stds)
            self.obs_data = utils.OtherUtils.zscore_normalization(self.obs_data, self.obs_means, self.obs_stds)
        elif NORMALIZATION == 'minmax':
            self.case_data = utils.OtherUtils.min_max_normalization(self.case_data, self.min, self.max)
            self.obs_data = utils.OtherUtils.min_max_normalization(self.obs_data, self.min, self.max)

        # 数据增强
        if DATA_ENHANCE:
            self.case_data = utils.CaseParser.data_enhance(self.case_data)
            self.obs_data = utils.ObsParser.data_enhance(self.obs_data)

        # 数据维度转换
        if DATA_FORMAT.startswith('grid'):
            if DATA_FORMAT.endswith('11'):
                self.case_data = utils.OtherUtils.map2grid(self.case_data, self.valid_grids, self.shape[0])
                self.obs_data = utils.OtherUtils.map2grid(self.obs_data, self.valid_grids, self.shape[0])
                self.obs_data = self.obs_data[:, 0: 1]
            elif DATA_FORMAT.endswith('33'):
                self.case_data, self.valid_grids = utils.OtherUtils.map2grid33(self.case_data, self.valid_grids, self.shape[0])
                self.obs_data = utils.OtherUtils.map2grid(self.obs_data, self.valid_grids, self.shape[0])
                self.obs_data = self.obs_data[:, 0: 1]

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]


class TestDataset(Dataset):
    def __init__(self, TEST_START_YEAR, TEST_END_YEAR, train_dataset):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA))
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TEST_START_YEAR, TEST_END_YEAR, AREA, train_dataset.months))
        self.obs_data = self.obs_data[:, 0: 1, :, :]    # 取第一个月
        self.shape = self.case_data.shape

        # 归一化
        if NORMALIZATION == 'zscore':
            self.case_data = utils.OtherUtils.zscore_normalization(
                self.case_data, train_dataset.case_means, train_dataset.case_stds)
            self.obs_data = utils.OtherUtils.zscore_normalization(
                self.obs_data, train_dataset.obs_means, train_dataset.obs_stds)
        elif NORMALIZATION == 'minmax':
            self.case_data = utils.OtherUtils.min_max_normalization(
                self.case_data, train_dataset.min, train_dataset.max)
            self.obs_data = utils.OtherUtils.min_max_normalization(
                self.obs_data, train_dataset.min, train_dataset.max)

        # 数据维度转换
        if DATA_FORMAT.startswith('grid'):
            if DATA_FORMAT.endswith('11'):
                self.case_data = utils.OtherUtils.map2grid(
                    self.case_data, train_dataset.valid_grids, self.shape[0])
                self.obs_data = utils.OtherUtils.map2grid(
                    self.obs_data, train_dataset.valid_grids, self.shape[0])
                self.obs_data = self.obs_data[:, 0: 1]
            elif DATA_FORMAT.endswith('33'):
                self.case_data, self.valid_grids = utils.OtherUtils.map2grid33(
                    self.case_data, train_dataset.valid_grids, self.shape[0])
                self.obs_data = utils.OtherUtils.map2grid(
                    self.obs_data, train_dataset.valid_grids, self.shape[0])
                self.obs_data = self.obs_data[:, 0: 1]

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]
