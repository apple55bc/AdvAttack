#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2019/11/12 21:55
@Author  : Apple QXTD
@File    : aa_cfg.py
@Desc:   :
"""

import os

join = os.path.join

MAIN_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MID_PATH = join(MAIN_PATH, 'mid_data')
MODEL_PATH = join(MAIN_PATH, 'model')
DATA_PATH = join(MAIN_PATH, 'data')
RESULT_PATH = join(MAIN_PATH, 'submit')

MAX_FOLD = 6


def _get_logger():
    import logging

    if not os.path.isdir(join(MAIN_PATH, 'logs')):
        os.makedirs(join(MAIN_PATH, 'logs'))
    LOG_PATH = join(MAIN_PATH, 'logs/log.txt')

    logging.basicConfig(filename=LOG_PATH,
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        filemode='w', datefmt='%Y-%m-%d%I:%M:%S %p')
    _logger = logging.getLogger(__name__)

    #  添加日志输出到控制台
    console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    _logger.addHandler(console)
    _logger.setLevel(logging.INFO)
    return _logger


def make_path():
    path_list = [
        MID_PATH,
        MODEL_PATH,
        RESULT_PATH
    ]
    for p in path_list:
        if not os.path.isdir(p):
            os.makedirs(p)


logger = _get_logger()


if __name__ == '__main__':
    make_path()
