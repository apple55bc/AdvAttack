#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/12 21:23
@File      :ec_cfg.py
@Desc      :
"""
from easydict import EasyDict as edict

config = edict()
config.max_seq_len = 32
config.num_hidden_layers = 4
config.label_num = 4
config.batch_size = 18
config.model_type = 'bert_roberta'