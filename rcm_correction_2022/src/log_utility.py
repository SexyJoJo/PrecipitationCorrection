# -*- encoding: utf-8 -*-
'''
@project :   C:\Project\radiation2022\radiation
@file    :   log_utility.py
@version :   1.0
@author  :   NUIST-LEE 
@time    :   2022-01-27 19:43:41
@description:  日志工具包
'''

# here put the import lib
import logging
import json
import logging.config
import os

 
def setup_logging(default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
    """
    @description: 初始化日志
    @param default_path:配置文件路径
    @param default_level:日志级别
    @param env_key:
    @return:
    """
    path = default_path
    value = os.getenv(env_key, None)
    # os.getenv(): Return the value of the environment variable varname if it exists,
    # or value if it doesn’t. value defaults to None.
    if value:
        path = value
    # print(os.getcwd(), '\n',
    #       os.path.realpath(__file__), '\n',
    #       os.path.split(os.path.realpath(__file__))[0], '\n',
    #       os.sys.path[0])
    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)

            for k, v in config.items():
                print(k, '\t', v)

            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

if __name__ == '__main__':
    pass
