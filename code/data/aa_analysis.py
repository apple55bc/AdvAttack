#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2019/11/12 21:56
@Author  : Apple QXTD
@File    : aa_analysis.py
@Desc:   :
"""

from data.aa_read import get_train
import numpy as np
from aa_cfg import *


def analysis_0():
    data_iter = get_train()

    id_num = 0
    eq_len = []
    neq_len = []
    avg_eq_len = []
    avg_neq_len = []

    for _, eqs_list, neqs_list in data_iter:
        id_num += 1
        avg_eq_len.append(len(eqs_list))
        avg_neq_len.append(len(neqs_list))
        for eq in eqs_list:
            eq_len.append(len(eq))
        for neq in neqs_list:
            neq_len.append(len(neq))
        print('\rnum: {}'.format(id_num), end='    ')

    print('\nover.')
    eq_len = np.array(eq_len)
    neq_len = np.array(neq_len)
    avg_eq_len = np.array(avg_eq_len)
    avg_neq_len = np.array(avg_neq_len)
    print('eq_len: ')
    print(eq_len.mean(), eq_len.max(), eq_len.min())
    print('neq_len: ')
    print(neq_len.mean(), neq_len.max(), neq_len.min())
    print('avg_eq_len: ')
    print(avg_eq_len.mean(), avg_eq_len.max(), avg_eq_len.min())
    print('avg_neq_len: ')
    print(avg_neq_len.mean(), avg_neq_len.max(), avg_neq_len.min())
    """
    num: 6036    
    over.
    eq_len: 
    20.5041782729805 71 5
    neq_len: 
    18.526652556971992 71 5
    avg_eq_len: 
    3.033300198807157 7 2
    avg_neq_len: 
    3.365970841616965 8 1
    """


def test_replace_word_percent():
    """测试文本中有多少比例的句子，存在同义词和反义词"""
    words = []
    with open(join(DATA_PATH, 'similar_words', 'similar.txt'), mode='r', encoding='utf-8') as fr:
        lines = [line.split(',') for line in fr.readlines()]
        for line in lines:
            words.extend(line)
    with open(join(DATA_PATH, 'similar_words', 'opposite.txt'), mode='r', encoding='utf-8') as fr:
        lines = [line.split(',') for line in fr.readlines()]
        for line in lines:
            words.extend(line)
    print('len of words: ', len(words))
    words = list(set(words))
    print('len of words: ', len(words))
    from error_correct.ec_get_text import get_line_text
    import jieba

    jieba.load_userdict(join(DATA_PATH, 'token_freq.txt'))
    jieba.load_userdict(join(DATA_PATH, 'law_word.txt'))

    all_num = 0
    in_num = 0
    line_iter = get_line_text()
    for s in line_iter:
        s_words = jieba.lcut(s)
        for w in s_words:
            if w in words:
                in_num += 1
                break
        all_num += 1
    print('{} / {} , {:.2f}%'.format(in_num, all_num, 100 * in_num / all_num))
    # 14668 / 38626 , 37.97%


def test_label_num():
    from data.aa_augmentation import DataAug
    from data.aa_read import trains_pairs
    import random
    import joblib

    data = joblib.load(join(MID_PATH, 'train_{}.joblib'.format(0)))
    data_aug = DataAug()
    mid_data = []
    for d in data:
        mid_data.append(trains_pairs(d))
    data = mid_data
    del mid_data
    labeled_data = []
    for piece in data:
        poses = piece[0]
        neges = piece[1]
        for pos in poses:
            labeled_data.append((pos, 1))
        for neg in neges:
            labeled_data.append((neg, 0))
    random.shuffle(labeled_data)

    num = 0
    for data, label in labeled_data:
        a = data[0]
        b = data[1]
        data_aug.trans_main(a, b, label)
        num += 1
        if num % 87 == 0:
            print('\rnum: {}'.format(num), end='     ')
        if num % 1000 == 999:
            s = input('continue? ')
            if s == 'q':
                data_aug.describe()
                break
            elif s == 'des':
                data_aug.describe()


if __name__ == '__main__':
    # analysis_0()
    # test_replace_word_percent()
    test_label_num()