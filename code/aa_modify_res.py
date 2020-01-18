#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/16 20:13
@File      :aa_modify_res.py
@Desc      :
"""
import pandas as pd
import numpy as np


def analysis():
    test = pd.read_csv('../submit/submit_20191217_203800.txt', sep='\t', header=None, index_col=None, encoding='utf-8')
    print(test.shape)
    test.columns = ['tip', 'label']
    values = test['label'].values
    print('values: {}'.format(len(values)))
    print('sum: {}'.format(values.sum()))

    print('test gate and sum: ')
    test = pd.read_csv('../submit/submit_score_20191217_203800.txt',
                       sep='\t', header=None, index_col=None, encoding='utf-8')
    print(test.shape)
    test.columns = ['tip', 'label']
    values = test['label'].values
    print('values: {}'.format(len(values)))
    values = np.sort(values)
    print(values[:10])
    for gate in range(100):
        gate_rate = gate / 100
        i = 0
        for i, v in enumerate(values):
            if v > gate_rate:
                break
        print('gate: {:.3f} per:{:.2f}% num: {} / {}  pos:{}'.format(
            gate_rate, i / len(values), i, len(values), len(values) - i))

def analysis2():
    test = pd.read_csv('../submit/submit_20191212_233539.txt', sep='\t', header=None, index_col=None, encoding='utf-8')
    print(test.shape)
    test.columns = ['tip', 'label']
    values = test['label'].values
    print('values: {}'.format(len(values)))
    print('sum: {}'.format(values.sum()))

    test = pd.read_csv('../submit/submit_20191216_093027.txt', sep='\t', header=None, index_col=None, encoding='utf-8')
    print(test.shape)
    test.columns = ['tip', 'label']
    values = test['label'].values
    print('values: {}'.format(len(values)))
    print('sum: {}'.format(values.sum()))


def trans():
    test = pd.read_csv('../submit/submit_score_20191217_203800.txt',
                       sep='\t', header=None, index_col=None, encoding='utf-8')
    test.columns = ['tip', 'label']
    test['label'] = test['label'].apply(lambda x: 1 if x > 0.49 else 0)
    values = test['label'].values
    print('values: {}'.format(len(values)))
    print('sum: {}'.format(values.sum()))
    test.to_csv('../submit/res_trans_49.txt', encoding='utf-8', index=False, sep='\t', header=False)


if __name__ == '__main__':
    # analysis()
    trans()
"""
gate: 0.450 per:0.82% num: 47964 / 58304  pos:10340
gate: 0.460 per:0.83% num: 48224 / 58304  pos:10080
gate: 0.470 per:0.83% num: 48472 / 58304  pos:9832
gate: 0.480 per:0.84% num: 48712 / 58304  pos:9592
gate: 0.490 per:0.84% num: 48933 / 58304  pos:9371
gate: 0.500 per:0.84% num: 49166 / 58304  pos:9138
gate: 0.510 per:0.85% num: 49366 / 58304  pos:8938
gate: 0.520 per:0.85% num: 49577 / 58304  pos:8727
gate: 0.530 per:0.85% num: 49794 / 58304  pos:8510
gate: 0.540 per:0.86% num: 50010 / 58304  pos:8294
gate: 0.550 per:0.86% num: 50228 / 58304  pos:8076
"""