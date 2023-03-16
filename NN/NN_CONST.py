# 数据根目录
CASE_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided case"
OBS_DIR = r"D:\PythonProject\PrecipitationCorrection\divide area\divided obs"

# 输出根目录
DESCRIPTION = "ANN-原始值-minmax-输入5个月3乘3-输出1个月1乘1"
LOSS_PATH = rf"./output/{DESCRIPTION}/loss"
MODEL_PATH = rf"./output/{DESCRIPTION}/models"
RESULT_PATH = rf"./output/{DESCRIPTION}/results"
EVALUATE_PATH = rf"./output/{DESCRIPTION}/评价指标"

DATE = "0302"  # 预报日期
CASE_NUM = "CASE1"  # CASE编号
TIME = "TIME00"  # 预报时次

# 流域名称 ['ChangJiang', 'HuangHe', 'HeHai']
BASIN = 'ChangJiang'

# 流域子区域简称
# 长江流域: 金沙江流域、 长江上游、 长江中下游
AREA = "JSJ"

# 数据集划分参数
TRAIN_START_YEAR = 1991
TRAIN_END_YEAR = 2019
DATA_ENHANCE = False
USE_ANOMALY = False
# JUMP_YEAR = 2019

# 训练参数
EPOCH = 150
BATCH_SIZE = 256
LR = 0.005

# MODEL_PATH = r"91-18.pth"
