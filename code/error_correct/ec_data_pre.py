#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/16 21:42
@File      :ec_data_pre.py
@Desc      :
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.aa_read import get_train
from aa_cfg import *
from bert4keras.utils import load_vocab


# 这个是获取已有文本中的词汇，用于纠错的候选词 correct_by_statistics will use it
def get_exist_words():
    import jieba
    from aa_cfg import join, DATA_PATH
    import json

    jieba.load_userdict(join(DATA_PATH, 'token_freq.txt'))
    jieba.load_userdict(join(DATA_PATH, 'law_word.txt'))

    chars = dict()

    train_iter = get_train()
    for _d in train_iter:
        _id, eqs_list, neqs_list = _d
        d_list = eqs_list
        d_list.extend(neqs_list)
        for s in d_list:
            s_list = jieba.lcut(s)
            for w in s_list:
                chars[w] = chars.get(w, 0) + 1
    chars = [(i, j) for i, j in chars.items() if j >= 10 and len(i) > 1]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]

    json.dump(
        chars,
        open(join(DATA_PATH, 'chars.dict'), 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


def simplify_vocab_dict():
    import joblib
    chars = dict()
    min_count = 1

    model_pre_save_path = join(MODEL_PATH, 'train_pre_for_error_detect')
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
    get_exist_words()
    simplify_vocab_dict()
