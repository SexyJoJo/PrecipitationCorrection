# 数据根目录
CASE_DIR = r"D:\PythonProjects\PrecipitationCorrection\divide area\divided case"
OBS_DIR = r"D:\PythonProjects\PrecipitationCorrection\divide area\divided obs"

DATE = "0302"  # 预报日期
CASE_NUM = "CASE1"  # CASE编号
TIME = "TIME00"  # 预报时次

# 预报月份 预报日期往后5个月

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

if USE_ANOMALY:
    # 输出根目录
    MODEL_PATH = "./距平值models"
    RESULT_PATH = "./距平值results"
    EVALUATE_PATH = "./距平值评价指标"
else:
    MODEL_PATH = "./原始值models"
    RESULT_PATH = "./原始值results"
    EVALUATE_PATH = "./原始值评价指标"
# JUMP_YEAR = 2019

# 训练参数
EPOCH = 50
BATCH_SIZE = 4
LR = 0.005
