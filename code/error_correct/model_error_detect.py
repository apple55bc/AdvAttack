#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/11 20:52
@File      :model_error_detect.py
@Desc      :
"""
import tensorflow as tf
import keras
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import json
from bert4keras.bert import BertModel, Bert4Seq2seq

from aa_cfg import *
from error_correct.ec_cfg import config as ec_cfg

layers = keras.layers
K = keras.backend


def load_pretrained_model(config_path,
                          checkpoint_file=None,
                          with_mlm=False,
                          seq2seq=False,
                          keep_words=None,
                          albert=False,
                          **kwargs):
    """根据配置文件和checkpoint文件来加载模型
    """
    _config = json.load(open(config_path))
    _config.update(kwargs)
    print(_config)

    if seq2seq:
        Bert = Bert4Seq2seq
    else:
        Bert = BertModel

    bert = Bert(vocab_size=_config['vocab_size'],
                max_position_embeddings=_config['max_position_embeddings'],
                hidden_size=_config['hidden_size'],
                num_hidden_layers=_config['num_hidden_layers'],
                num_attention_heads=_config['num_attention_heads'],
                intermediate_size=_config['intermediate_size'],
                hidden_act=_config['hidden_act'],
                dropout_rate=_config['hidden_dropout_prob'],
                embedding_size=_config.get('embedding_size'),
                with_mlm=with_mlm,
                keep_words=keep_words,
                block_sharing=albert)

    bert.build()

    if checkpoint_file is not None:
        bert.load_weights_from_checkpoint(checkpoint_file)

    return bert.model


def testloss(y_true, y_pred):  # 目标y_pred需要是one hot形式
    loss = tf.reduce_mean(y_true - y_pred, axis=-1)
    # loss = tf.reduce_mean(loss, axis=-1)
    return loss  # 即log(分子/分母)


class DetectModel:
    def __init__(self, keep_words=None):
        l_input_ids = layers.Input(shape=(ec_cfg.max_seq_len,), dtype='int32')
        l_token_type_ids = layers.Input(shape=(ec_cfg.max_seq_len,), dtype='int32')

        bert_model = load_pretrained_model(join(DATA_PATH, ec_cfg.model_type, 'bert_config.json'),
                                           # join(DATA_PATH, ec_cfg.model_type, 'bert_model.ckpt'),
                                           None,
                                           num_hidden_layers=ec_cfg.num_hidden_layers,
                                           keep_words=keep_words)  # 建立模型，加载权重
        net = bert_model([l_input_ids, l_token_type_ids])

        # net = layers.Embedding(10000, 10)(l_input_ids)
        # net = layers.Dense(config.label_num, activation='sigmoid', name='class')(net)
        self._crf = CRF(ec_cfg.label_num, test_mode='viterbi')
        net = self._crf(net)
        # print('net after crf', net)
        self.model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=net)
        self.model.summary()

    def compile(self):
        opt = keras.optimizers.Adam(learning_rate=1e-5)
        self.model.compile(optimizer=opt, loss=crf_loss, metrics=[crf_viterbi_accuracy])

if __name__ == '__main__':
    # dm = DetectModel()
    load_pretrained_model(join(DATA_PATH, 'bert/bert_config.json'),
                          join(DATA_PATH, 'bert/bert_model.ckpt'),
                          num_hidden_layers=2)  # 建立模型，加载权重