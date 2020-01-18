#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/11/16 9:58
@Author  : Apple QXTD
@File    : aa_base_bert_train.py
@Desc:   :
"""
import numpy as np
import joblib
import random
import keras
import pandas as pd

random.seed(322)
np.random.seed(322)

from aa_cfg import *
from data.aa_read import trains_pairs
# import uniout
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer
from keras.callbacks import Callback
from keras.optimizers import Adam
from focal_loss import focal_loss
from data.aa_augmentation import DataAug


config_path = os.path.join(DATA_PATH, 'bert_roberta/bert_config.json')
# checkpoint_path = os.path.join(DATA_PATH, 'bert_roberta/bert_model.ckpt')
checkpoint_path = None
# config_path = os.path.join(DATA_PATH, 'bert_wwm/bert_config.json')
# checkpoint_path = os.path.join(DATA_PATH, 'bert_wwm/bert_model.ckpt')
batch_size = 4
steps_per_epoch = 1000
epochs = 40
init_epoch = 0
max_set_len = 70

token_dict = joblib.load(join(MODEL_PATH, 'train_pre', 'token_dict.joblib'))
keep_words = joblib.load(join(MODEL_PATH, 'train_pre', 'keep_words.joblib'))

tokenizer = SimpleTokenizer(token_dict)  # 建立分词器
train_len = 0
eva_len = 0
data_aug = DataAug()


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def one_hot(vec, depth: int):
    _res = np.zeros((len(vec), depth))
    for i, v in enumerate(vec):
        _res[i, int(v)] = 1
    return _res


def get_test():
    _test = pd.read_csv(join(DATA_PATH, 'test_set.csv'), sep='\t')
    # 纠错
    logger.info('Apply corrector ...')
    from error_correct.detect_and_correct import get_correct_fn
    corrector = get_correct_fn()
    _test['question1'] = _test['question1'].apply(corrector)
    _test['question2'] = _test['question2'].apply(corrector)
    del corrector
    keras.backend.clear_session()

    _trains_x = list()
    _trains_s = list()
    questions_1 = _test['question1'].values
    questions_2 = _test['question2'].values
    for i in range(_test.shape[0]):
        q1, q2 = questions_1[i], questions_2[i]
        x, s = tokenizer.encode(q1[:max_set_len - 3], q2[:max_set_len - 3])
        _trains_x.append(x)
        _trains_s.append(s)
        # transed_examples.append([x, s])
    _test = _test[['qid']]
    _test['label'] = 0
    return _test, padding(_trains_x), padding(_trains_s)


def data_generator(fold=0, is_train=True):
    global train_len
    global eva_len
    logger.info('generator pre ...   info fold: {} is_train: {}'.format(fold, is_train))
    if is_train:
        data = joblib.load(join(MID_PATH, 'train_{}.joblib'.format(fold)))
    else:
        data = joblib.load(join(MID_PATH, 'val_{}.joblib'.format(fold)))
    if is_train:
        mid_data = []
        for d in data:
            mid_data.append(trains_pairs(d))
        data = mid_data
        del mid_data

    # 把正反例对，转换为有标签的对
    # print('data: ')
    # print(data[0])
    # print(len(data[0]))
    labeled_data = []
    for piece in data:
        poses = piece[0]
        neges = piece[1]
        for pos in poses:
            labeled_data.append((pos, 1))
        for neg in neges:
            labeled_data.append((neg, 0))

    random.shuffle(labeled_data)
    if is_train:
        train_len = len(labeled_data)
    else:
        eva_len = len(labeled_data)
    logger.info('generator pre over.  data num: {}'.format(len(labeled_data)))

    _num = 0
    while True:
        X, S, L = [], [], []
        for data, label in labeled_data:
            _num += 1
            # a, b, label = data[0], data[1], label
            if not is_train or _num <= 0:
                a, b, label = data[0], data[1], label
            else:
                a, b, label = data_aug.trans_main(data[0], data[1], label)
            a = a[:max_set_len - 3]
            b = b[:max_set_len - 3]
            x, s = tokenizer.encode(a, b)
            X.append(x)
            S.append(s)
            L.append(label)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                L = one_hot(L, 2)
                yield [X, S], L
                X, S, L = [], [], []


class LogRecord(keras.callbacks.Callback):
    def __init__(self):
        super(LogRecord, self).__init__()
        self._save_step = 0

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        logger.info('epoch: {} loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}'.format(
            epoch, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy']))

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
        self._save_step += 1
        if self._save_step % 100 == 0:
            logger.info('step: {}  loss: {} '.format(self._save_step, logs['loss']))


test, trains_x, trains_s = get_test()


def train(fold=0, only_predict=False, need_val=True):
    # from accum_optimizer import AccumOptimizer

    if fold in []:
        only_predict = True
    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        seq2seq=False,
        keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    )
    x_in = keras.Input(shape=(None,), name='Token')
    s_in = keras.Input(shape=(None,), name='Segment')
    output = model([x_in, s_in])
    output = keras.layers.core.Lambda(lambda x: x[:, 0, :])(output)
    output = keras.layers.Dense(2, activation='sigmoid')(output)
    model = keras.Model([x_in, s_in], output)

    opt = Adam(5e-6)
    model.compile(opt, loss=[focal_loss(alpha=0.85)], metrics=['accuracy'])

    if fold == 0:
        model.summary()

    save_dir = join(MODEL_PATH, 'bert_res/out_{}'.format(fold))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not only_predict and init_epoch == 0:
        for l in os.listdir(save_dir):  # 删空
            os.remove(join(save_dir, l))
    save_path = join(save_dir, 'trained.ckpt')

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        save_path, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='min', period=1)
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=save_dir, histogram_freq=0, write_graph=False,
        write_grads=False, update_freq=320)
    # weight_decay_callback = keras.callbacks.LearningRateScheduler(
    #     schedule=lambda epoch, lr: lr * (epochs - epoch) / epochs if epoch > 0 else 1e-6
    # )

    if only_predict:
        model.load_weights(save_path)
    else:
        if init_epoch > 0:
            logger.info('Continue train. Load weight...')
            model.load_weights(save_path)
        model.fit_generator(
            data_generator(fold, True),
            validation_data=data_generator(fold, False),
            validation_steps=100,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2,
            # workers=3,
            class_weight=None,
            initial_epoch=init_epoch,
            callbacks=[
                checkpoint_callback,
                tensorboard_callback,
                # weight_decay_callback,
                LogRecord()
            ]
        )
    if need_val:
        logger.info('evaluate...')
        if only_predict:  # 这个执行一次，用于重新统计 eva_len
            next(data_generator(fold, False))
        eva_result = model.evaluate_generator(data_generator(fold, False), steps=int(eva_len / batch_size))
    else:
        eva_result = []
    logger.info('predict...')
    results = model.predict([trains_x, trains_s], batch_size=batch_size)
    print('result shape: {}'.format(results.shape))
    assert len(results) == test.shape[0]

    keras.backend.clear_session()
    return results[:, 1], eva_result


if __name__ == '__main__':
    import datetime

    logger.info('checkpoint path: {}'.format(checkpoint_path))
    logger.info('Max fole: {}'.format(MAX_FOLD))

    eva_results = []
    for _fold in range(MAX_FOLD):
        logger.info('=' * 20)
        logger.info('>>>>>>>  fold: {}'.format(_fold))
        res, eva_res = train(_fold, only_predict=True, need_val=False)
        test['label'] += res
        logger.info('eva result: {}'.format(eva_res))
        eva_results.append(eva_res)
    test['label'] /= MAX_FOLD
    test.to_csv(join(RESULT_PATH, "submit_score_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"),
               encoding='utf-8', index=False, sep='\t', header=False)
    test['label'] = test['label'].apply(lambda x: 1 if x > 0.5 else 0)
    # test.to_csv(join(RESULT_PATH, "submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"),
    #            encoding='utf-8', index=False, sep='\t', header=False)
    test.to_csv(join(RESULT_PATH, "result.txt"),
               encoding='utf-8', index=False, sep='\t', header=False)
    logger.info('Save over. Final eva mean: {}'.format(np.mean(np.array(eva_results), axis=0)))
