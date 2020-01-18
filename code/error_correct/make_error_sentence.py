#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/11 19:55
@File      :make_error_sentence.py
@Desc      :制作错别字样本
"""
import pinyin
import pandas as pd
import numpy as np
import random
import jieba
import re
import joblib

from aa_cfg import *

jieba.load_userdict(join(DATA_PATH, 'token_freq.txt'))


def mprint(_s, end='\n'):
    # print(_s, end=end)
    pass


def dprint(_s, end='\n'):
    # print(_s, end=end)
    pass


class ErrorMaker:
    def __init__(self, load=True):
        save_dir = join(MODEL_PATH, 'error_maker_save')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if load and os.path.exists(os.path.join(save_dir, 'words.df')) and \
                os.path.exists(os.path.join(save_dir, 'pinyin_dis.df')):
            self._words = joblib.load(os.path.join(save_dir, 'words.df'))
            self._pinyin_dis = joblib.load(os.path.join(save_dir, 'pinyin_dis.df'))
            self._pinyin_dis = self._pinyin_dis[self._pinyin_dis['edit_dis'] <= 1]
            self._pinyin_comb = list(set(self._words['pinyin'].values))
            mprint('loaded. words: {}  pinyin_dis: {}'.format(self._words.shape, self._pinyin_dis.shape))
            mprint('columns of words: {}'.format(self._words.columns))
            mprint('columns of pinyin_dis: {}'.format(self._pinyin_dis.columns))
            mprint('kinds of pinyin combination: {}'.format(len(self._pinyin_comb)))
        else:
            txt_path = join(DATA_PATH, 'token_freq.txt')
            mprint('ErrorMaker==========')
            mprint('read ...')
            self._words = pd.read_csv(txt_path, header=None, index_col=None, sep=' ')
            self._words.columns = ['word', 'freq', 'prop']
            mprint('readed. shape: {}'.format(self._words.shape))
            mprint('apply ...')
            self._words['pinyin'] = self._words['word'].apply(
                lambda s: pinyin.get(s, format="strip", delimiter=" "))  # numerical, strip, diacritical
            # 删除不识别的拼音
            mprint('del unknown ...')
            pattern = re.compile('^[a-z ]+$')
            self._words['tag'] = self._words['pinyin'].apply(lambda _s: 1 if pattern.match(_s) else 0)
            self._words = self._words[self._words['tag'] == 1]
            del self._words['tag']
            mprint('del over. shape: {}'.format(self._words.shape))
            all_pinyin = self._words['pinyin'].values
            pinyin_set = []
            for pin in all_pinyin:
                pinyin_set.extend(pin.split(' '))
            pinyin_set = list(set(pinyin_set))
            mprint('set length: {}'.format(len(pinyin_set)))
            pinyin_dis = pd.DataFrame(pinyin_set, columns=['pinyin'])
            pinyin_dis['tag'] = 1
            pinyin_df_r = pinyin_dis.copy()
            pinyin_df_r.rename(columns={'pinyin': 'sim_pinyin'}, inplace=True)
            pinyin_dis = pd.merge(pinyin_dis, pinyin_df_r, on='tag')
            del pinyin_dis['tag'], pinyin_df_r
            mprint('merge shape: {}'.format(pinyin_dis.shape))
            pinyin_dis['edit_dis'] = pinyin_dis.apply(
                lambda df: self.edit_distance(df['pinyin'], df['sim_pinyin']), axis=1)
            pinyin_dis = pinyin_dis[pinyin_dis['edit_dis'] <= 2]
            mprint(pinyin_dis.shape)
            mprint(pinyin_dis['edit_dis'].describe())
            self._pinyin_dis = pinyin_dis[pinyin_dis['edit_dis'] > 0]
            joblib.dump(self._words, os.path.join(save_dir, 'words.df'))
            joblib.dump(self._pinyin_dis, os.path.join(save_dir, 'pinyin_dis.df'))
            self._pinyin_comb = list(set(self._words['pinyin'].values))
            self._pinyin_dis = self._pinyin_dis[self._pinyin_dis['edit_dis'] <= 1]
        # 加载停用词
        # with open(join(DATA_PATH, 'stop_words.txt'), mode='r', encoding='utf-8') as fr:
        #     self.stop_words = [w.strip() for w in fr.readlines()]
        random.shuffle(self._pinyin_comb)
        self._shuffle_count = 0  # 查询调用超过20次就重新随机排序一次
        mprint('init over.')

    def create_ill_sentence(self, sentence, no_replace_rate=0.1):
        """返回（病句，[起始位置]，[长度]，[正确词]）"""
        words = jieba.lcut(sentence)
        begins = []
        lengths = []
        correct_words = []
        dprint('len of sentence: {}'.format(len(words)))
        if len(words) <= 1 or random.random() < no_replace_rate:  # 只有一个词不做替换，并且有一定概率永远不做替换
            return sentence, begins, lengths, correct_words, sentence
        error_n = random.randint(1, max(1, int(len(words) * 0.15)))
        indexes = list(np.arange(len(words)))
        random.shuffle(indexes)
        replace_indexes = []
        for i in range(error_n):
            replace_word = self.select_similar_word(words[indexes[i]])
            if replace_word == '':
                continue
            replace_indexes.append(indexes[i])  # begins最后再统计
            correct_words.append(words[indexes[i]])
            lengths.append(len(replace_word))
            words[indexes[i]] = replace_word
        for b in replace_indexes:
            _index = 0
            for i in range(b):
                _index += len(words[i])
            begins.append(_index)
        return ''.join(words), begins, lengths, correct_words, sentence

    def select_similar_word(self, word):
        candidates = self._select_candidates(word)
        need_drop_n = False
        if word in ['我', '我们', '他', '他们', '她', '她们', '你', '你们', '什么']:
            need_drop_n = True
        res = ''
        for cand_comb in candidates:
            cand_df = self._words[self._words['pinyin'] == cand_comb]
            if need_drop_n:
                cand_df = cand_df[~cand_df['prop'].isin(['n', 'nr', 'nrt'])]
            cand_df = cand_df[cand_df['word'] != word]
            if cand_df.shape[0] == 0:
                continue
            res = random.choices(cand_df['word'].values.tolist(), weights=cand_df['freq'].values.tolist())[0]
        dprint(res)
        return res

    def _select_candidates(self, word):
        candidates = []  # 这个候选按照顺序选取，因为有的拼音可能找不到
        _pinyin_list = pinyin.get(word, format="strip", delimiter=" ").split(' ')
        if random.random() < 0.75:  # 和原词拼音相同的概率
            dprint('pinyin add init')
            comb = ' '.join(_pinyin_list)
            if comb in self._pinyin_comb:
                candidates.append(comb)
        if random.random() < 0.5:  # 随机替换其中一个拼音
            _index = random.randint(0, len(_pinyin_list) - 1)
            similar_pinyin = self._pinyin_dis[self._pinyin_dis['pinyin'] == _pinyin_list[_index]]['sim_pinyin'].values
            similar_pinyin = list(similar_pinyin)
            random.shuffle(similar_pinyin)
            for _sp in similar_pinyin:
                this_comb = _pinyin_list.copy()
                this_comb[_index] = _sp
                comb = ' '.join(this_comb)
                if comb in self._pinyin_comb:
                    dprint('pinyin replace one')
                    candidates.append(comb)
                    if len(candidates) >= 2:
                        return candidates
        if random.random() < 0.2 and len(_pinyin_list) >= 2:  # 随机替换其中两个拼音
            _indexes = random.sample(np.arange(len(_pinyin_list)).tolist(), 2)
            similar_1 = list(
                self._pinyin_dis[self._pinyin_dis['pinyin'] == _pinyin_list[_indexes[0]]]['sim_pinyin'].values)
            similar_2 = list(
                self._pinyin_dis[self._pinyin_dis['pinyin'] == _pinyin_list[_indexes[1]]]['sim_pinyin'].values)
            random.shuffle(similar_1)
            random.shuffle(similar_2)
            for _sp1 in similar_1:
                for _sp2 in similar_2:
                    this_comb = _pinyin_list.copy()
                    this_comb[_indexes[0]] = _sp1
                    this_comb[_indexes[1]] = _sp2
                    comb = ' '.join(this_comb)
                    if comb in self._pinyin_comb:
                        dprint('pinyin replace two')
                        candidates.append(comb)
                        if len(candidates) >= 2:
                            return candidates
        if random.random() < 0.5 and len(_pinyin_list) >= 2:  # 随机删除一个拼音
            this_comb = _pinyin_list.copy()
            this_comb.pop(random.randint(0, len(this_comb) - 1))
            comb = ' '.join(this_comb)
            if comb in self._pinyin_comb:
                dprint('pinyin del one')
                candidates.append(comb)
                if len(candidates) >= 2:
                    return candidates
        # 随机在左侧或者右侧增加拼音
        dprint('pinyin add')
        init_comb = ' '.join(_pinyin_list)
        self._shuffle_count += 1
        if self._shuffle_count > 1000:
            random.shuffle(self._pinyin_comb)
            self._shuffle_count = 0
        for _pc in self._pinyin_comb:
            if _pc != init_comb:
                _pc = str(_pc)
                if random.random() < 0.5:  # 左侧。下面后半部分abs逻辑限制增加最多一个音节
                    if _pc.endswith(init_comb) and abs(len(_pc.split(' ')) - len(_pinyin_list)) <= 1:
                        candidates.append(_pc)
                        if len(candidates) >= 2:
                            return candidates
                else:
                    if _pc.startswith(init_comb) and abs(len(_pc.split(' ')) - len(_pinyin_list)) <= 1:
                        candidates.append(_pc)
                        if len(candidates) >= 2:
                            return candidates
        return candidates

    @staticmethod
    def edit_distance(word1, word2):
        len1 = len(word1)
        len2 = len(word2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if word1[i - 1] == word2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return int(dp[len1][len2])


if __name__ == '__main__':
    error = ErrorMaker()
    while True:
        s = input('q to quit: \n')
        if s == 'q':
            break
        print(error.create_ill_sentence(s))
