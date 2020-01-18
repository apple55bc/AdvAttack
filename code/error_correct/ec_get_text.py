#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/12 21:03
@File      :ec_get_text.py
@Desc      :
"""
from data.aa_read import get_train
from error_correct.ec_cfg import config as ec_cfg


def get_line_text():
    train_iter = get_train()
    for _d in train_iter:
        _id, eqs_list, neqs_list = _d
        d_list = eqs_list
        d_list.extend(neqs_list)
        for s in d_list:
            yield s[:ec_cfg.max_seq_len]
