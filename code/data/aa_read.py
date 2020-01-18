#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2019/11/12 22:25
@Author  : Apple QXTD
@File    : aa_read.py
@Desc:   :
"""
from xml.dom import minidom
from aa_cfg import *
import random

random.seed(211)


def get_train():
    data = minidom.parse(join(DATA_PATH, 'train_set.xml'))
    collection = data.documentElement
    questions = collection.getElementsByTagName("Questions")
    for question in questions:
        _id = question.getAttribute("number")
        eqs = question.getElementsByTagName('EquivalenceQuestions')[0].getElementsByTagName('question')
        neqs = question.getElementsByTagName('NotEquivalenceQuestions')[0].getElementsByTagName('question')
        eqs_list = []
        for eq in eqs:
            if eq.firstChild is not None:
                eqs_list.append(eq.firstChild.data)
        neqs_list = []
        for neq in neqs:
            if neq.firstChild is not None:
                neqs_list.append(neq.firstChild.data)
        yield _id, eqs_list, neqs_list


def trains_pairs(data, top_n=None):
    _, eqs_list, neqs_list = data
    pos_pairs = []
    if len(eqs_list) > 1:
        for l in eqs_list:
            for r in eqs_list[1:]:
                if l != r:
                    pos_pairs.append([l, r])
    neg_pairs = []
    for l in eqs_list:
        for r in neqs_list:
            neg_pairs.append([l, r])
            neg_pairs.append([r, l])
    random.shuffle(pos_pairs)
    random.shuffle(neg_pairs)
    if top_n is not None:
        return pos_pairs[:top_n], neg_pairs[:top_n]
    else:
        return pos_pairs, neg_pairs


def fix_utf_8():
    with open(join(DATA_PATH, 'train_set.xml'), mode='r', encoding='utf-8') as fr:
        all_train_data = fr.readlines()
        all_train_data[0] = all_train_data[0].replace('utf8', 'utf-8')
    with open(join(DATA_PATH, 'train_set.xml'), mode='w', encoding='utf-8') as fw:
        fw.writelines(all_train_data)


if __name__ == '__main__':
    fix_utf_8()
