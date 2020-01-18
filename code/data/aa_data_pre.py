#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Time    : 2019/11/16 9:24
@Author  : Apple QXTD
@File    : aa_data_pre.py
@Desc:   :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.aa_read import get_train, trains_pairs
import random
random.seed(171111)
import joblib
from aa_cfg import *
from bert4keras.utils import load_vocab


def split_data(split_n=MAX_FOLD):
    # 5个交叉验证集
    print('read...')
    data = list(get_train())
    print('shuffle...')
    random.shuffle(data)
    val_len = int(len(data) / split_n)
    for i in range(split_n):
        if i == split_n -1 :
            val = data[val_len * i:]
        else:
            val = data[val_len * i:val_len * (i + 1)]
        train = data[:val_len * i]
        train.extend(data[val_len * (i + 1) :])
        val_final = []
        for d in val:
            val_final.append(trains_pairs(d, 2))
        random.shuffle(val_final)
        print('save {}'.format(i))
        joblib.dump(train, join(MID_PATH, 'train_{}.joblib'.format(i)))
        joblib.dump(val_final, join(MID_PATH, 'val_{}.joblib'.format(i)))


def simplify_vocab_dict():
    import json
    chars = dict()

    min_count = 1

    model_pre_save_path = join(MODEL_PATH, 'train_pre')
    if not os.path.isdir(model_pre_save_path):
        os.makedirs(model_pre_save_path)

    data = get_train()
    for _, pos, neg in data:
        for sentence in pos:
            for w in sentence:
                chars[w] = chars.get(w, 0) + 1
        for sentence in neg:
            for w in sentence:
                chars[w] = chars.get(w, 0) + 1

    chars = [(i, j) for i, j in chars.items() if j >= min_count]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]
    json.dump(
        chars,
        open(join(model_pre_save_path, 'chars.dict'), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    # checkpoint_path = os.path.join(main_path, 'model/bert/bert_model.ckpt')
    dict_path = os.path.join(DATA_PATH, 'bert_roberta/vocab.txt')

    _token_dict = load_vocab(dict_path)  # 读取词典
    token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

    for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

    for c in chars:
        if c in _token_dict:
            token_dict[c] = len(token_dict)
            keep_words.append(_token_dict[c])
    print('len of keep_words: ', len(keep_words))
    joblib.dump(token_dict, join(model_pre_save_path, 'token_dict.joblib'))
    joblib.dump(keep_words, join(model_pre_save_path, 'keep_words.joblib'))


if __name__ == '__main__':
    split_data()
    simplify_vocab_dict()