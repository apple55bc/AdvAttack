#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/12 21:17
@File      :augmentation.py
@Desc      :
"""
from aa_cfg import *
import pandas as pd
from error_correct.make_error_sentence import ErrorMaker
import jieba
import random

random.seed(2111)

jieba.load_userdict(join(DATA_PATH, 'token_freq.txt'))
jieba.load_userdict(join(DATA_PATH, 'law_word.txt'))


class DataAug:
    def __init__(self):
        similar = pd.read_csv(join(DATA_PATH, 'similar_words/similar.txt'), header=None, index_col=None,
                              encoding='utf-8')
        similar.columns = ['word_a', 'word_b']
        deny = pd.read_csv(join(DATA_PATH, 'similar_words/similar.txt'), header=None, index_col=None,
                              encoding='utf-8')
        deny.columns = ['word_a', 'word_b']
        opposite = pd.read_csv(join(DATA_PATH, 'similar_words/opposite.txt'), header=None, index_col=None,
                               encoding='utf-8')
        opposite.columns = ['word_a', 'word_b']
        # 不替换长词的反义词
        opposite['len'] = opposite['word_a'].apply(lambda x: len(x))
        opposite = opposite[opposite['len'] <= 2]
        del opposite['len']

        similar_r = similar.copy()
        similar_r.columns = ['word_b', 'word_a']
        similar = pd.concat([similar, similar_r], axis=0, sort=False)
        del similar_r
        deny_r = deny.copy()
        deny_r.columns = ['word_b', 'word_a']
        deny = pd.concat([deny, deny_r], axis=0, sort=False)
        del deny_r
        opposite_r = opposite.copy()
        opposite_r.columns = ['word_b', 'word_a']
        opposite = pd.concat([opposite, opposite_r], axis=0, sort=False)
        del opposite_r
        self.similar = similar
        self.opposite = opposite
        self.deny = deny
        with open(join(DATA_PATH, 'stop_words.txt'), mode='r', encoding='utf-8') as fr:
            self.stop_words = [line.strip() for line in fr.readlines() if line != '']
        self.deny_words = [
            '未', '没有', '无', '非'
        ]
        self.error_maker = ErrorMaker()
        self._statistics = {
            'num': 0,
            'label_ori_1': 0,
            'label_trans': 0,
            'label_0': 0,
            'label_1': 0,
        }

    def trans_main(self, sa, sb, label: int):
        if label == 1:
            self._statistics['label_ori_1'] += 1
        if random.random() < 0.3:
            return self._return(sa, sb, label)
        # 双否替换， 不改变标签
        if random.random() < 0.15:
            sa, sb, tip = self.trans_deny_both(sa, sb)
            if tip:
                return self._return(sa, sb, label)
        # 选择哪个句子进行替换
        if random.random() < 0.5:
            is_a = True
            rep_s = sa
        else:
            is_a = False
            rep_s = sb
        # 对一个句子进行替换
        rep_s, label_trans = self.trans_sentence(rep_s, label)
        if label != label_trans:
            self._statistics['label_trans'] += 1
        if is_a:
            return self._return(rep_s, sb, label_trans)
        else:
            return self._return(sa, rep_s, label_trans)

    def _return(self, sa, sb, label: int):
        self._statistics['num'] += 1
        if label == 0:
            self._statistics['label_0'] += 1
        elif label == 1:
            self._statistics['label_1'] += 1
        else:
            logger.warning('>' * 40)
            logger.warning('Label is not 0 or 1 !')
        return sa, sb, label

    def trans_sentence(self, rep_s, label: int):
        tip = False
        if random.random() < 0.6:  # 实际上只有37.97%的句子中包含同义词反义词
            if random.random() < 0.7:
                rep_s, tip = self.trans_similar(rep_s)
                if not tip and label == 1:
                    rep_s, tip = self.trans_opposite(rep_s)
                    if tip: label = 1 - label
            elif label == 1:
                rep_s, tip = self.trans_opposite(rep_s)
                if tip:
                    label = 1 - label
                else:
                    rep_s, tip = self.trans_similar(rep_s)
            if tip:
                return rep_s, label
        # 单句否定
        if label == 1 and random.random() < 0.35:
            rep_s, tip = self.trans_deny(rep_s)
            if tip:
                return rep_s, 1 - label
        # 制造错句
        if random.random() < 0.5:
            rep_s, tip = self.trans_to_error(rep_s)
            if tip:
                return rep_s, label
        # 剩下的随机替换词顺序
        rep_s, tip = self.trans_words_position(rep_s)
        return rep_s, label

    def trans_to_error(self, sentence):
        error_text, begins, lengths, _, _ = self.error_maker.create_ill_sentence(sentence, no_replace_rate=0.0)
        self._statistics['trans_to_error'] = self._statistics.get('trans_to_error', 0) + 1
        return error_text, len(begins) > 0

    def trans_similar(self, sentence):
        sentence_l = jieba.lcut(sentence)
        s_df = self.similar[self.similar['word_a'].isin(sentence_l)]
        if s_df.shape[0] == 0:
            return sentence, False
        replace_index = random.randint(0, s_df.shape[0] - 1)
        word_a = s_df['word_a'].values[replace_index]
        word_b = s_df['word_b'].values[replace_index]
        sentence = sentence.replace(word_a, word_b, 1)
        self._statistics['trans_similar'] = self._statistics.get('trans_similar', 0) + 1
        return sentence, True

    def trans_opposite(self, sentence):
        sentence_l = jieba.lcut(sentence)
        o_df = self.opposite[self.opposite['word_a'].isin(sentence_l)]
        if o_df.shape[0] == 0:
            return sentence, False
        replace_index = random.randint(0, o_df.shape[0] - 1)
        word_a = o_df['word_a'].values[replace_index]
        word_b = o_df['word_b'].values[replace_index]
        sentence = sentence.replace(word_a, word_b, 1)
        self._statistics['trans_opposite'] = self._statistics.get('trans_opposite', 0) + 1
        return sentence, True

    def trans_deny(self, sentence):
        exist_w = []
        for w in self.deny['word_a'].values:
            if w in sentence:
                exist_w.append(w)
        if len(exist_w) == 0:
            return self.trans_deny_force(sentence)
        random.shuffle(exist_w)
        o_df = self.deny[self.deny['word_a'] == exist_w[0]]
        replace_index = random.randint(0, o_df.shape[0] - 1)
        word_a = o_df['word_a'].values[replace_index]
        word_b = o_df['word_b'].values[replace_index]
        sentence = sentence.replace(word_a, word_b, 1)
        self._statistics['trans_deny'] = self._statistics.get('trans_deny', 0) + 1
        return sentence, True

    def trans_deny_force(self, sentence):
        # 分词，过滤停用词，在特殊词汇之前添加否定词语
        sen_l = jieba.lcut(sentence)
        sen_l = [w for w in sen_l if w not in self.stop_words]
        if len(sen_l) == 0:
            return sentence, False
        rep_ori_w = random.sample(sen_l, 1)[0]
        rep_w = random.sample(self.deny_words, 1)[0]
        sentence = sentence.replace(rep_ori_w, rep_ori_w + rep_w, 1)
        self._statistics['trans_deny_force'] = self._statistics.get('trans_deny_force', 0) + 1
        return sentence, True

    def trans_deny_force_both(self, sen_0, sen_1):
        # 分词，过滤停用词，在特殊词汇之前添加否定词语
        sen_l_0 = jieba.lcut(sen_0)
        sen_l_0 = [w for w in sen_l_0 if w not in self.stop_words]
        sen_l_1 = jieba.lcut(sen_1)
        sen_l_1 = [w for w in sen_l_1 if w not in self.stop_words]
        sen_l = list(set(sen_l_0).intersection(sen_l_1))
        if len(sen_l) == 0:
            return sen_0, sen_1, False
        rep_ori_w = random.sample(sen_l, 1)[0]
        rep_w = random.sample(self.deny_words, 1)[0]
        sen_0 = sen_0.replace(rep_ori_w, rep_ori_w + rep_w, 1)
        sen_1 = sen_1.replace(rep_ori_w, rep_ori_w + rep_w, 1)
        self._statistics['trans_deny_force_both'] = self._statistics.get('trans_deny_force_both', 0) + 1
        return sen_0, sen_1, True

    def trans_deny_both(self, sen_0, sen_1):
        exist_w = []
        for w in self.deny['word_a'].values:
            if w in sen_0 and w in sen_1:
                exist_w.append(w)
        if len(exist_w) == 0:
            return self.trans_deny_force_both(sen_0, sen_1)
        random.shuffle(exist_w)
        o_df = self.deny[self.deny['word_a'] == exist_w[0]]
        replace_index = random.randint(0, o_df.shape[0] - 1)
        word_a = o_df['word_a'].values[replace_index]
        word_b = o_df['word_b'].values[replace_index]
        sen_0 = sen_0.replace(word_a, word_b, 1)
        sen_1 = sen_1.replace(word_a, word_b, 1)
        self._statistics['trans_deny_both'] = self._statistics.get('trans_deny_both', 0) + 1
        return sen_0, sen_1, True

    def trans_words_position(self, sentence):
        sen_l = jieba.lcut(sentence)
        if len(sen_l) <= 1:
            return sentence, False
        # 替换的距离不超过2个单词
        rep_dis = random.randint(1, min(2, len(sen_l) - 1))
        rep_begin = random.randint(0, len(sen_l) - rep_dis - 1)
        mid_word = sen_l[rep_begin]
        sen_l[rep_begin] = sen_l[rep_begin + rep_dis]
        sen_l[rep_begin + rep_dis] = mid_word
        self._statistics['trans_words_position'] = self._statistics.get('trans_words_position', 0) + 1
        return ''.join(sen_l), True

    def describe(self):
        for key, item in self._statistics.items():
            print('key: {}  value: {}'.format(key, item))


if __name__ == '__main__':
    data_aug = DataAug()
    print(data_aug.trans_sentence('在家里自己做吃的然后通过微信卖了这样犯法吗', 1))
    print(data_aug.trans_sentence('食品经营者应当如何贮存食品', 1))
    print(data_aug.trans_sentence('实习生工作的时候受伤可以认定为工伤吗', 1))
    print(data_aug.trans_sentence('申请人身安全保护令保护期限多长', 1))
    print(data_aug.trans_sentence('被告人是否可以委托辩护人为自己辩护', 1))
    print(data_aug.trans_sentence('依法应当注册的保健食品', 1))
    print(data_aug.trans_sentence('在校生去兼职上班遇车祸受伤，司机要承担责任吗？', 1))
    print(data_aug.trans_sentence('小区高空抛物，砸伤人怎么办', 1))
    print(data_aug.trans_sentence('适用“因感情不和分居满二年的”规定的有什么地方值得注意', 1))
    print(data_aug.trans_sentence('婚前财产公证双方都需要到场吗', 1))
    print(data_aug.trans_sentence('未经允许拆除邻居的违章建筑需赔偿吗', 1))
