#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@project :   RCM_correction
@file    :   utils_stat.py
@version :   1.0
@author  :   NUIST-LEE
@time    :   2023-2-7 15:15
@description:
https://www.cnblogs.com/marszhw/p/12175454.html
计算统计度量
'''

# here put the import lib
# import random
import math

# 计算平均值
def mean(x):
    return sum(x) / len(x)


# 计算每一项数据与均值的差
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


# 辅助计算函数 dot product 、sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


# 方差
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    # return sum_of_squares(deviations) / (n - 1)
    return sum_of_squares(deviations) / n


# 标准差
def standard_deviation(x):
    return math.sqrt(variance(x))


# 协方差
def covariance(x, y):
    n = len(x)
    # return dot(de_mean(x), de_mean(y)) / (n - 1)
    return dot(de_mean(x), de_mean(y)) / n


# 相关系数
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0

def f01_test():
    # a = [random.randint(0, 10) for t in range(20)]
    # b = [random.randint(0, 10) for t in range(20)]
    a = [1, 2, 3]
    b = [30, 20, 10]
    print(a)
    print(b)
    print(standard_deviation(a))
    print(standard_deviation(b))
    print(correlation(a, b))

if __name__ == '__main__':
    f01_test()
    print('done')
