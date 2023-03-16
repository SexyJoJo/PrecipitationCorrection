#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   RCM_correction
@file    :   p09gridsearch.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-2-8 18:18
@description:
20230208，以一年数据为测试数据，建模搜索最佳模型结构。
20230208-搜索ANN的结构。
'''

# here put the import lib
import os
import torch
import torch.utils.data
import numpy as np
from p04trainandtest import AreaModelTrainTest
import random
import logging
import log_utility

# 固定随机种子
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)    # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(SEED)    # 为所有GPU设置随机种子


# 超参数集中定义
class Const_Param(object):
    layer_num = [2, 3]
    hidden_num = [32, 64, 128]
    activation = ['tanh', 'relu', 'sigmoid']
    batchnorm_flag = [False, True]     # True:使用batchnorm；False：不使用batchnorm
    dropout_flag = [False, True]    # True：使用dropout，False：不使用dropout
    lat_lon_ix = 2  # 输入数据包含经纬度数据，则为2；不包含经纬度数据，则为0


    # 单例模式代码
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    def __init__(self):
        pass

    # 固定随机种子
    @classmethod
    def fix_seed(self, SEED = 123):
        # SEED = 123
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)    # 为当前GPU设置随机种子
        # torch.cuda.manual_seed_all(SEED)    # 为所有GPU设置随机种子


def grid_seearch():
    for layer in Const_Param.layer_num:
        for hidden in Const_Param.hidden_num:
            for act in Const_Param.activation:
                for bn in Const_Param.batchnorm_flag:
                    for dp in Const_Param.dropout_flag:
                        tt = AreaModelTrainTest(test_year=1991)
                        tt.num_layers = layer
                        tt.hidden_dim = hidden
                        tt.activation = act
                        tt.batchnormflag = bn
                        tt.droupoutflag = dp
                        tt.epochs = 300
                        s = f'ANN-l{layer}h{hidden}-{act}-' \
                            f'bn{1 if bn else 0}-' \
                            f'dp{1 if dp else 0}'
                        logging.info(f'begin: {s}')

                        tt.organize_data()
                        tt.train_test()








def z01test():
    print(Const_Param.layer_num)







if __name__ == '__main__':
    logconfigjson = os.path.join(os.sys.path[0], 'logging.json')
    log_utility.setup_logging(default_path=logconfigjson)
    # z01test()
    Const_Param.fix_seed()

    grid_seearch()


    print('done')
