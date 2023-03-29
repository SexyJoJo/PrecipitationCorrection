import os

from torch.utils.data import Dataset
from NN_CONST import *
import torch
import utils

CASE_DIR = os.path.join(CASE_DIR, DATE, CASE_NUM, TIME, BASIN)
OBS_DIR = os.path.join(OBS_DIR, BASIN)


class TrainDataset(Dataset):
    def __init__(self, JUMP_YEAR):
        super().__init__()
        self.case_data = torch.Tensor(
            utils.CaseParser.get_many_2d_pravg(CASE_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, JUMP_YEAR))
        self.shape = self.case_data.shape
        self.months = utils.OtherUtils.get_predict_months(DATE, self.shape[1])
        self.obs_data = torch.Tensor(
            utils.ObsParser.get_many_2d_pravg(OBS_DIR, TRAIN_START_YEAR, TRAIN_END_YEAR, AREA, self.months, JUMP_YEAR))

        # 归一化
        self.min = torch.min(self.case_data)
        self.max = torch.max(self.case_data)
        self.case_data = utils.OtherUtils.min_max_normalization(self.case_data, self.min, self.max)
        self.obs_data = utils.OtherUtils.min_max_normalization(self.obs_data, self.min, self.max)

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

        # 归一化
        self.case_data = utils.OtherUtils.min_max_normalization(self.case_data, train_dataset.min, train_dataset.max)
        self.obs_data = utils.OtherUtils.min_max_normalization(self.obs_data, train_dataset.min, train_dataset.max)

    def __getitem__(self, index):
        return self.case_data[index], self.obs_data[index]

    def __len__(self):
        return self.case_data.shape[0]
