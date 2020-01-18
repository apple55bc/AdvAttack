#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/11 21:39
@File      :train_errror_detect.py
@Desc      :
"""
import numpy as np
import keras
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aa_cfg import *
from error_correct.ec_cfg import config
from error_correct.ec_get_text import get_line_text
from bert4keras.utils import SimpleTokenizer
from error_correct.model_error_detect import DetectModel
from error_correct.make_error_sentence import ErrorMaker
import joblib
import random

random.seed(1111)
np.random.seed(1111)

error_maker = ErrorMaker()
token_print = True

token_dict = joblib.load(join(MODEL_PATH, 'train_pre_for_error_detect', 'token_dict.joblib'))
tokenizer = SimpleTokenizer(token_dict)
keep_words = joblib.load(join(MODEL_PATH, 'train_pre_for_error_detect', 'keep_words.joblib'))
model = DetectModel(keep_words=keep_words)
model.compile()


def one_hot(vec, depth: int):
    res = np.zeros((len(vec), depth))
    for i, v in enumerate(vec):
        res[i, int(v)] = 1
    return res


def get_eva_text():
    line_iter = get_line_text()
    logger.info('load outer data mlc_dataset_part_ac_3 ...')
    num = 0
    while True:
        num += 1
        try:
            line = (next(line_iter))
        except StopIteration:
            line_iter = get_line_text()
            continue
        error_text, begins, lengths, _, _ = error_maker.create_ill_sentence(line)
        text_tokens = tokenizer.tokenize(error_text, False, False)[:config.max_seq_len - 2]
        tokens = list()
        tokens.append("[CLS]")
        for token in text_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        input_ids = [token_dict[c] for c in tokens]
        while len(input_ids) < config.max_seq_len:
            input_ids.append(0)
        yield tokens, [input_ids], np.zeros_like([input_ids], dtype=np.int)


def generator():
    global token_print

    line_iter = get_line_text()
    X, Y = [], []
    while True:
        try:
            line = next(line_iter)
        except StopIteration:
            line_iter = get_line_text()
            continue
        error_text, begins, lengths, _, _ = error_maker.create_ill_sentence(line)
        text_tokens = tokenizer.tokenize(error_text, False, False)[:config.max_seq_len - 2]

        tokens = list()
        tokens.append("[CLS]")
        for token in text_tokens:
            tokens.append(token)
        tokens.append("[SEP]")
        input_ids = [token_dict[c] for c in tokens]
        # Zero-pad up to the sequence length.
        while len(input_ids) < config.max_seq_len:
            input_ids.append(0)

        label = np.zeros(config.max_seq_len, dtype=np.int)
        for begin, length in zip(begins, lengths):
            if begin + length + 2 > config.max_seq_len:
                if begin + 2 > config.max_seq_len:
                    continue
                else:
                    length = config.max_seq_len - 2 - begin
            if length == 1:  # S = 3
                label[begin + 1] = 3
            else:
                label[begin + 1: begin + 1 + length] = 1
                label[begin + length] = 2
        label = one_hot(label, 4)

        if token_print:
            token_print = False
            print("*** Example ***")
            print("tokens: %s" % " ".join(tokens))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("label[:3]: %s" % np.array(label[:3]))

        if len(input_ids) != config.max_seq_len:
            print(input_ids)
            print(len(input_ids))
            print(config.max_seq_len)
            raise ValueError('????')
        X.append(np.array(input_ids, dtype=np.int))
        Y.append(label)
        if len(X) >= config.batch_size:
            # print('shape X : ', np.array(X).shape)
            # print('shape Y : ', np.array(Y).shape)
            yield [np.array(X), np.zeros_like(np.array(X), dtype=np.int)], np.array(Y, dtype=np.float32)
            X, Y = [], []


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10
        self.eva_iter = get_eva_text()

    def on_epoch_end(self, epoch, logs=None):
        for _ in range(3):
            tokens, ids, segs = next(self.eva_iter)
            res = model.model.predict([ids, segs])[0]
            print('init: {}'.format(''.join(tokens)))
            res_str = ''
            for i, r in enumerate(res):
                if np.argmax(r) > 0:
                    try:
                        res_str += str(tokens[i])
                    except IndexError:
                        break
                else:
                    res_str += '_'
            print(res_str)


if __name__ == '__main__':
    save_path = join(MODEL_PATH, 'detect')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        join(save_path, 'weights.hdf5'), monitor='loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='min', period=1)
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0, patience=5, verbose=0, mode='min',
        baseline=None, restore_best_weights=True)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=save_path, histogram_freq=0, write_graph=False,
        write_grads=False, update_freq=320)

    if os.path.exists(join(save_path, 'weights.hdf5')):
        print('Load from saved model....')
        model.model.load_weights(join(save_path, 'weights.hdf5'))
    model.model.fit_generator(
        generator(),
        steps_per_epoch=200,
        epochs=25,
        verbose=1,
        class_weight=None,
        callbacks=[
            checkpoint_callback,
            # early_stop_callback,
            tensorboard_callback,
            Evaluate()
        ]
    )
