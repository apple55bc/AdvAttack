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
# import uniout
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer
from keras.optimizers import Adam
from focal_loss import focal_loss
from data.aa_augmentation import DataAug

# config_path = os.path.join(DATA_PATH, 'bert_roberta/bert_config.json')
# checkpoint_path = os.path.join(DATA_PATH, 'bert_roberta/bert_model.ckpt')
config_path = os.path.join(DATA_PATH, 'bert_wwm/bert_config.json')
checkpoint_path = os.path.join(DATA_PATH, 'bert_wwm/bert_model.ckpt')
batch_size = 4
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


def get_dev():
    _ori_dev = pd.read_csv(join(DATA_PATH, 'test_sample.csv'), header=0, index_col=None, encoding='utf-8')
    _ori_dev.columns = ['a', 'b', 'c', 'label_a', 'label_b']
    # 纠错
    logger.info('Apply corrector ...')
    from error_correct.detect_and_correct import get_correct_fn
    corrector = get_correct_fn()
    _ori_dev['a'] = _ori_dev['a'].apply(corrector)
    _ori_dev['b'] = _ori_dev['b'].apply(corrector)
    _ori_dev['c'] = _ori_dev['c'].apply(corrector)
    del corrector
    keras.backend.clear_session()

    questions_1 = _ori_dev['a'].values.tolist()
    questions_2 = _ori_dev['b'].values.tolist()
    labels = [1] * len(questions_1)
    questions_1.extend(_ori_dev['a'].values.tolist())
    questions_2.extend(_ori_dev['c'].values.tolist())
    labels += _ori_dev['label_b'].values.tolist()
    questions_1.extend(_ori_dev['b'].values.tolist())
    questions_2.extend(_ori_dev['c'].values.tolist())
    labels += _ori_dev['label_b'].values.tolist()
    _dev = pd.DataFrame(np.array([questions_1, questions_2, labels]).T, columns=['question1', 'question2', 'label'])

    _trains_x = list()
    _trains_s = list()
    for _i in range(len(questions_1)):
        q1, q2 = questions_1[_i], questions_2[_i]
        x, s = tokenizer.encode(q1[:max_set_len - 3], q2[:max_set_len - 3])
        _trains_x.append(x)
        _trains_s.append(s)
        # transed_examples.append([x, s])
    return _dev, padding(_trains_x), padding(_trains_s)


dev, trains_x, trains_s = get_dev()


def predict(fold=0):
    # from accum_optimizer import AccumOptimizer

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

    save_dir = join(MODEL_PATH, 'bert_wwm_aug_focal/out_{}'.format(fold))
    # save_dir = join(MODEL_PATH, 'bert_res/out_{}'.format(fold))
    save_path = join(save_dir, 'trained.ckpt')

    model.load_weights(save_path)
    logger.info('predict...')
    results = model.predict([trains_x, trains_s], batch_size=batch_size)
    print('result shape: {}'.format(results.shape))
    assert len(results) == dev.shape[0]

    keras.backend.clear_session()
    return results[:, 1]


if __name__ == '__main__':
    logger.info('checkpoint path: {}'.format(checkpoint_path))
    logger.info('Max fole: {}'.format(MAX_FOLD))
    MAX_FOLD = 5

    dev['label_predcit'] = 0.0
    dev['label'] = dev['label'].astype(int)
    for _fold in range(MAX_FOLD):
        logger.info('=' * 20)
        logger.info('>>>>>>>  fold: {}'.format(_fold))
        res = predict(_fold)
        # print('fold: {}  res: {}'.format(_fold, res[:10]))
        dev['label_predcit'] += res

    dev['label_predcit'] /= MAX_FOLD
    dev['label_predcit'] = dev['label_predcit'].apply(lambda x: 1 if x > 0.5 else 0)
    dev['accu'] = (dev['label_predcit'] == dev['label']) * 1
    print('Accu: {:.5f}'.format(dev['accu'].mean()))
    dev = dev[dev['accu'] == 0]
    for i in range(dev.shape[0]):
        print('l: {}  q1: {} >>> q2: {}'.format(dev['label'].values[i], dev['question1'].values[i],
                                                dev['question2'].values[i]))

"""
Roberta focal-0.85
Accu: 0.76667
l: 1  q1: 出车祸后，保险公司拒赔该怎么办 >>> q2: 上了保险，出了车祸，保险公司不赔钱，咋办
l: 1  q1: 毒驾撞死人会判刑吗？怎么判刑 >>> q2: 吸度后开车撞死人会判刑吗？怎么判刑
l: 1  q1: 出了交通事故不赔钱需要坐牢吗 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 撤销权的意思是什么 >>> q2: 什么是撒销权
l: 1  q1: 保险公司可以拒赔非医保用药费用吗 >>> q2: 请教下，保险公司拒赔韭医保用药费用合不合法
l: 0  q1: 夫妇解除同居关系，债务怎么分配？ >>> q2: 恋人解除同居关系，债务怎么分配？
l: 0  q1: 逃逸后的肇事司机又选择了自首，会从轻处理吗 >>> q2: 肇事司机逃逸后被抓，会从轻处理吗
l: 0  q1: 我把车借给了朋友，朋友交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任
l: 1  q1: 交通事故中被判赔偿，如果不赔会不会坐牢 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 什么是撤销权 >>> q2: 什么是撒销权
l: 1  q1: 保险公司拒赔非医保用药费用合不合法 >>> q2: 请教下，保险公司拒赔韭医保用药费用合不合法
l: 1  q1: 超速的认定标准是什么？咋扣分 >>> q2: 超速的认定标准是什么？咋减分
l: 0  q1: 肇事司机逃逸后自首，会从轻处理吗 >>> q2: 肇事司机逃逸后被抓，会从轻处理吗
l: 0  q1: 朋友借我的车，交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任

Bert-wwm 强化 focal-0.85
Accu: 0.78333
l: 1  q1: 毒驾撞死人会判刑吗？怎么判刑 >>> q2: 吸度后开车撞死人会判刑吗？怎么判刑
l: 1  q1: 出了交通事故不赔钱需要坐牢吗 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 撤销权的意思是什么 >>> q2: 什么是撒销权
l: 1  q1: 保险公司可以拒赔非医保用药费用吗 >>> q2: 请教下，保险公司拒赔韭医保用药费用合不合法
l: 0  q1: 夫妇解除同居关系，债务怎么分配？ >>> q2: 恋人解除同居关系，债务怎么分配？
l: 0  q1: 我把车借给了朋友，朋友交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任
l: 1  q1: 交通事故护理费的计算标准是什么 >>> q2: 交通事故日常护理费的计算标准是什么
l: 1  q1: 交通事故中被判赔偿，如果不赔会不会坐牢 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 如果是在工作日加班，那工资应该咋算呢 >>> q2: 如果是在工作日架班，那工资应该咋算呢？
l: 1  q1: 什么是撤销权 >>> q2: 什么是撒销权
l: 1  q1: 保险公司拒赔非医保用药费用合不合法 >>> q2: 请教下，保险公司拒赔韭医保用药费用合不合法
l: 0  q1: 夫妻解除同居关系，债务怎么分配？ >>> q2: 恋人解除同居关系，债务怎么分配？
l: 0  q1: 朋友借我的车，交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任

Bert-wwm 强化 focal-0.85 纠错
Accu: 0.81667
l: 1  q1: 出了交通事故不赔钱需要坐牢吗 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 撤销权的意思是什么 >>> q2: 什么是倾销权
l: 1  q1: 保险公司可以拒赔非医保用工费用吗 >>> q2: 请教下，保险公司拒赔就医保用药费用合不合法
l: 0  q1: 夫妇解除同居关系，债务怎么分配？ >>> q2: 恋人解除同居关系，债务怎么分配？
l: 0  q1: 我把车借给了朋友，朋友交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任
l: 1  q1: 交通事故护理费的计算标准是什么 >>> q2: 交通事故日常护理费的计算标准是什么
l: 1  q1: 交通事故中被判赔偿，如果不赔会不会坐牢 >>> q2: 交通事故中被判赔偿，如果不赔会不会蹲号子
l: 1  q1: 什么是撤销权 >>> q2: 什么是倾销权
l: 1  q1: 保险公司拒赔非医保用工费用合不合法 >>> q2: 请教下，保险公司拒赔就医保用药费用合不合法
l: 0  q1: 夫妻解除同居关系，债务怎么分配？ >>> q2: 恋人解除同居关系，债务怎么分配？
l: 0  q1: 朋友借我的车，交通违章了，该由谁承担责任 >>> q2: 朋友借我他的车，交通违章了，该由谁承担责任
"""