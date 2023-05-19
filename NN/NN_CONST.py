# 数据根目录
CASE_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided case"
OBS_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided obs"

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
DATA_ENHANCE = 0  # 数据增强以增加样本量，仅针对二维
NORMALIZATION = 'zscore'  # [minmax, zscore]
DATA_FORMAT = 'map'  # [map, grid11, gird33]
MODEL = 'LSTM11'  # [LSTM_CNN, ANN, ANN_h16, ANN33, LSTM, LSTM11, UNet]

# 训练参数
EARLY_STOP = False
EPOCH = 400
if DATA_FORMAT == 'map' and DATA_ENHANCE:
    BATCH_SIZE = 32
elif DATA_FORMAT == 'map':
    BATCH_SIZE = 8
elif DATA_FORMAT.startswith('grid'):
    BATCH_SIZE = 1024
else:
    raise '请配置正确的数据格式'

LR = 0.001

# 输出根目录
# DESCRIPTION = "LSTM2_CNN1-原始值-zscore-lr.005-最小损失模型epoch150-输入n月输出n月"
# DESCRIPTION = "LSTM-原始值-zscore-最小损失模型-输入n月1乘1输出1月1乘1-batch1024-epoch150-.001lr"
DESCRIPTION = f"{MODEL}-{'早停模型' if EARLY_STOP else '最终模型'}-验证测试分开-输入{DATA_FORMAT}" \
              f"{'增强'+str(DATA_ENHANCE*8+1)+'倍' if DATA_ENHANCE else ''}-" \
              f"batch{BATCH_SIZE}-epoch{EPOCH}-{LR}lr"
LOSS_PATH = rf"./output/{DESCRIPTION}/loss"
MODEL_PATH = rf"./output/{DESCRIPTION}/models"
RESULT_PATH = rf"./output/{DESCRIPTION}/results"
EVALUATE_PATH = rf"./output/{DESCRIPTION}/评价指标"
