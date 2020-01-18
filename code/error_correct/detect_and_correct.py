#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/13 14:25
@File      :detect_and_correct.py
@Desc      :
"""
from bert4keras.utils import SimpleTokenizer
from error_correct.model_error_detect import DetectModel
from error_correct.correct_by_statistics import Statistics
import joblib
from aa_cfg import *
from error_correct.ec_cfg import config as ec_cfg
import numpy as np


def get_correct_fn():
    save_path = join(MODEL_PATH, 'detect')

    token_dict = joblib.load(join(MODEL_PATH, 'train_pre_for_error_detect', 'token_dict.joblib'))
    tokenizer = SimpleTokenizer(token_dict)
    keep_words = joblib.load(join(MODEL_PATH, 'train_pre_for_error_detect', 'keep_words.joblib'))
    model = DetectModel(keep_words=keep_words)
    model.compile()
    model.model.load_weights(join(save_path, 'weights.hdf5'))

    checker = Statistics()

    def correct(error_text):
        text_tokens = tokenizer.tokenize(error_text, False, False)[:ec_cfg.max_seq_len - 2]
        tokens = list()
        tokens.append("[CLS]")
        for token in text_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        input_ids = [token_dict[c] if c in token_dict.keys() else token_dict['[UNK]'] for c in tokens]
        while len(input_ids) < ec_cfg.max_seq_len:
            input_ids.append(0)

        seg_ids = np.zeros_like(input_ids, dtype=np.int)

        ids, segs = [input_ids], [seg_ids]
        res = model.model.predict([ids, segs])[0][1:-1]

        begins_pred = []
        lengths_pred = []
        this_len = 0
        for i, r in enumerate(res):
            if np.argmax(r) > 0:
                if this_len == 0:
                    begins_pred.append(i)
                this_len += 1
            else:
                if this_len > 0:
                    lengths_pred.append(this_len)
                    this_len = 0
        else:
            if this_len > 0:
                lengths_pred.append(this_len)

        res_str = checker.correct(error_text, begins_pred, lengths_pred)

        return res_str

    return correct


if __name__ == '__main__':
    _correct = get_correct_fn()
    print(_correct('夫妇解除同居关系，债务怎么分配？'))
    print(_correct('吸毒后开车撞死人会判刑吗？怎么判刑'))
    print(_correct('如果是在工作日架班，那工资应该咋算呢？'))
    print(_correct('谁有权分配获得车火死亡赔偿金'))
    print(_correct('上呢了保险，出了车祸，保险公司不赔钱，蚱么办？'))
    print(_correct('什么是撒销权'))
    print(_correct('没有经过邻居同意就拆除他的胃胀建筑需要赔偿吗'))
    print(_correct('同居时女方生下孩子，男方不负责壬抚养怎么办？'))