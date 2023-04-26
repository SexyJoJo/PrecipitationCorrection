# 数据根目录
CASE_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided case"
OBS_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided obs"

# 输出根目录
# DESCRIPTION = "LSTM2_CNN1-原始值-zscore-lr.005-最小损失模型epoch150-输入n月输出n月"
# DESCRIPTION = "ANN-原始值-zscore-最小损失模型-输入n月3乘3输出1月1乘1-.0005lr-去dropout"
DESCRIPTION = "LSTM-原始值-zscore-最小损失模型-输入n月1乘1输出1月1乘1-batch1024-epoch150-.001lr"
LOSS_PATH = rf"./output/{DESCRIPTION}/loss"
MODEL_PATH = rf"./output/{DESCRIPTION}/models"
RESULT_PATH = rf"./output/{DESCRIPTION}/results"
EVALUATE_PATH = rf"./output/{DESCRIPTION}/评价指标"

DATE = "0131"  # 预报日期
CASE_NUM = "CASE1"  # CASE编号
TIME = "TIME00"  # 预报时次

# 流域名称 ['ChangJiang', 'HuangHe', 'HeHai']
BASIN = 'ChangJiang'

# 流域子区域简称
# 长江流域: 金沙江流域、 长江上游、 长江中下游
AREA = "JSJ"

# 数据集划分参数
TRAIN_START_YEAR = 1991
TRAIN_END_YEAR = 2018

# 数据处理参数
NORMALIZATION = 'zscore'    # [minmax, zscore]
DATA_FORMAT = 'grid11'  # [map, grid11, gird33]on
MODEL = 'LSTM11'      # [LSTM_CNN, ANN, ANN33, LSTM, LSTM11]

# 训练参数
EPOCH = 150
BATCH_SIZE = 1024
LR = 0.001