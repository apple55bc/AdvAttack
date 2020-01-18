#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author    :apple.li
@Time      :2019/12/11 19:40
@File      :correct_by_statictic.py
@Desc      :基于上下文统计的纠错
"""
from aa_cfg import *
from error_correct.ec_get_text import get_line_text
import jieba
import jieba.posseg
import pinyin
import joblib
import numpy as np
import math
import json

jieba.load_userdict(join(DATA_PATH, 'token_freq.txt'))
jieba.load_userdict(join(DATA_PATH, 'law_word.txt'))

replace_type = ['m', 'nr', 'ns', 'nt', 'nz', 'r', 'x', 'w']
dict_save_path = join(MODEL_PATH, 'error_maker_save')
if not os.path.isdir(dict_save_path):
    os.makedirs(dict_save_path)


class Statistics:
    def __init__(self, model_path=join(dict_save_path, 'all_dicts.dict'), ensure_len=True):
        self._ensure_len = ensure_len
        self._replace_type = replace_type
        self._all_dicts = joblib.load(model_path)
        self._len_weights = [1.0, 2.0, 3.0]  # 上下文匹配序列长度对应的权重。
        with open(join(DATA_PATH, 'law_word.txt'), mode='r', encoding='utf-8') as fr:
            self.law_word = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
        exist_chars = json.load(open(join(DATA_PATH, 'chars.dict'), mode='r', encoding='utf-8'))
        self.law_word.extend(exist_chars)
        with open(join(DATA_PATH, 'token_freq.txt'), mode='r', encoding='utf-8') as fr:
            others = [line.strip().split(' ')[:2] for line in fr.readlines() if len(line.strip()) != 0]
            others = sorted(others, key=lambda x: -int(x[1]))
            others = [x[0] for x in others]
            self.law_word.extend(others[:2000])

    def correct(self, sentence, begins, lengths):
        assert len(begins) == len(lengths), '{} {}'.format(begins, lengths)
        if len(begins) == 0:
            return sentence
        res = ''
        for i, begin, length in zip(range(len(begins)), begins, lengths):
            if i > 0:
                left_sentence = sentence[begins[i - 1] + lengths[i - 1]:begin]
            else:
                left_sentence = sentence[:begin]
                res += left_sentence
            if i < len(begins) - 1:
                right_sentence = sentence[begin + length:begins[i + 1]]
            else:
                right_sentence = sentence[begin + length:]
            error_word = sentence[begin:begin + length]
            res += self.correct_single_word(left_sentence, right_sentence, error_word)
            res += right_sentence
        return res

    def correct_single_word(self, left, right, word):
        law_word, tip = self._correct_by_law_word(word)
        if tip:
            return law_word
        left = jieba.posseg.lcut(left.strip())
        right = jieba.posseg.lcut(right.strip())
        left_list = []
        right_list = []
        for i in range(len(left)):
            if left[i].flag in self._replace_type:
                left_list.append(left[i].flag)
            else:
                left_list.append(left[i].word)
        for i in range(len(right)):
            if right[i].flag in self._replace_type:
                right_list.append(right[i].flag)
            else:
                right_list.append(right[i].word)
        word_pinyin = self._get_pinyin(word)
        res_dict = dict()
        if len(left_list) > 1:
            self._add_dict(res_dict, self._lr_1_score(left_list[-1], word_pinyin))
            if len(left_list) > 2:
                self._add_dict(res_dict, self._lr_2_score(left_list[-2], left_list[-1], word_pinyin))
                if len(left_list) > 3:
                    self._add_dict(
                        res_dict, self._lr_3_score(left_list[-3], left_list[-2], left_list[-1], word_pinyin))
        if len(right_list) > 1:
            self._add_dict(res_dict, self._rl_1_score(right_list[0], word_pinyin))
            if len(right_list) > 2:
                self._add_dict(res_dict, self._rl_2_score(right_list[1], right_list[0], word_pinyin))
                if len(right_list) > 3:
                    self._add_dict(
                        res_dict, self._rl_3_score(right_list[2], right_list[1], right_list[0], word_pinyin))
        res_list = [(key, value) for key, value in res_dict.items()]
        res_list = sorted(res_list, key=lambda x: x[1], reverse=True)
        res_list = [w[0] for w in res_list]
        if len(res_list) == 0:
            return law_word
        # 如果存在，查看编辑距离是否在1以内，不是则优先替换law_word
        for w in res_list[:10]:
            if self.edit_distance(w, word) <= 1:
                return w
        return law_word

    def _correct_by_law_word(self, word):
        candidates = [w for w in self.law_word if len(word) == len(w)]
        candidates = [w for w in candidates if self.edit_distance(w, word) <= 1]
        if len(candidates) > 0:
            word_pinyin = self._get_pinyin(word)
            for cand in candidates:
                if self._get_pinyin(cand) == word_pinyin:
                    return cand, True
            return candidates[0], False
        else:
            return word, False

    def _add_dict(self, res, add_dict):
        for key in add_dict.keys():
            res[key] = res.get(key, 0) + add_dict[key]
        return res

    def _lr_1_score(self, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['lr_1_dict'].get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[0])

    def _lr_2_score(self, w2, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['lr_2_dict'].get(w2, {}).get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[1])

    def _lr_3_score(self, w3, w2, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['lr_3_dict'].get(w3, {}).get(w2, {}).get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[2])

    def _rl_1_score(self, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['rl_1_dict'].get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[0])

    def _rl_2_score(self, w2, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['rl_2_dict'].get(w2, {}).get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[1])

    def _rl_3_score(self, w3, w2, w1, word_pinyin: list):
        candidate_dict = self._all_dicts['rl_3_dict'].get(w3, {}).get(w2, {}).get(w1, {}).copy()
        return self._reset_weights(candidate_dict, word_pinyin, self._len_weights[2])

    def _reset_weights(self, candidate_dict: dict, word_pinyin, len_weight):
        except_list = list()
        for key in candidate_dict.keys():
            candidate_dict[key] = math.log(candidate_dict[key] + 1) * \
                                  self._cal_pinyin_dis(self._get_pinyin(key), word_pinyin) * len_weight
            if self._ensure_len:
                if len(key) != len(word_pinyin):
                    except_list.append(key)
        if len(except_list) > 0:
            for key in except_list:
                candidate_dict.pop(key)
        return candidate_dict

    @staticmethod
    def _get_pinyin(word):
        return pinyin.get(word, format="strip", delimiter=" ").split(' ')

    def _cal_pinyin_dis(self, left: list, right: list):
        if len(left) > len(right):
            mid = right
            right = left
            left = mid
        max_score = 0.0
        for l in range(len(right) - len(left) + 1):
            score_piece = 0.0
            for i in range(len(left)):
                score_piece += max(2.0 - self.edit_distance(left[i], right[l + i]), 0.0)
            max_score = max(max_score, score_piece)
        return max_score / len(right)

    @staticmethod
    def edit_distance(word1, word2):
        if word1 == word2:
            return 0
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


"""
        'lr_1_dict': lr_1_dict,
        'lr_2_dict': lr_2_dict,
        'lr_3_dict': lr_3_dict,
        'rl_1_dict': rl_1_dict,
        'rl_2_dict': rl_2_dict,
        'rl_3_dict': rl_3_dict
"""


def cal_fre_dicts(max_num=300000):
    line_iter = get_line_text()
    line_index = 0
    lr_1_dict = dict()
    lr_2_dict = dict()
    lr_3_dict = dict()
    rl_1_dict = dict()
    rl_2_dict = dict()
    rl_3_dict = dict()
    while True:
        line_index += 1
        if line_index % 87 == 0:
            print('\rline index: {} / {}'.format(line_index, max_num), end='   ')
        if line_index > max_num:
            break
        try:
            line = next(line_iter)
        except StopIteration:
            print('\nline index: {} / {}'.format(line_index, max_num), end='   ')
            print('StopIteration')
            break
        line_cuted = jieba.posseg.lcut(line.strip())
        for i, x in enumerate(line_cuted):
            if x.flag in ['w', 'm', 'x']:
                continue
            end_n = len(line_cuted) - i - 1
            begin_n = i
            # 分别加入6个长度的词典中
            if end_n >= 1:
                w1 = line_cuted[i + 1].word
                if line_cuted[i + 1].flag in replace_type:
                    w1 = line_cuted[i + 1].flag
                rl_1_dict[w1] = rl_1_dict.get(w1, {})
                rl_1_dict[w1][x.word] = rl_1_dict[w1].get(x.word, 0) + 1
                if end_n >= 2:
                    w2 = line_cuted[i + 2].word
                    if line_cuted[i + 2].flag in replace_type:
                        w2 = line_cuted[i + 2].flag
                    rl_2_dict[w2] = rl_2_dict.get(w2, {})
                    rl_2_dict[w2][w1] = rl_2_dict[w2].get(w1, {})
                    rl_2_dict[w2][w1][x.word] = rl_2_dict[w2][w1].get(x.word, 0) + 1
                    if end_n >= 3:
                        w3 = line_cuted[i + 3].word
                        if line_cuted[i + 3].flag in replace_type:
                            w3 = line_cuted[i + 3].flag
                        rl_3_dict[w3] = rl_3_dict.get(w3, {})
                        rl_3_dict[w3][w2] = rl_3_dict[w3].get(w2, {})
                        rl_3_dict[w3][w2][w1] = rl_3_dict[w3][w2].get(w1, {})
                        rl_3_dict[w3][w2][w1][x.word] = rl_3_dict[w3][w2][w1].get(x.word, 0) + 1
            # 左边部分
            if begin_n >= 1:
                w1 = line_cuted[i - 1].word
                if line_cuted[i - 1].flag in replace_type:
                    w1 = line_cuted[i - 1].flag
                lr_1_dict[w1] = lr_1_dict.get(w1, {})
                lr_1_dict[w1][x.word] = lr_1_dict[w1].get(x.word, 0) + 1
                if begin_n >= 2:
                    w2 = line_cuted[i - 2].word
                    if line_cuted[i - 2].flag in replace_type:
                        w2 = line_cuted[i - 2].flag
                    lr_2_dict[w2] = lr_2_dict.get(w2, {})
                    lr_2_dict[w2][w1] = lr_2_dict[w2].get(w1, {})
                    lr_2_dict[w2][w1][x.word] = lr_2_dict[w2][w1].get(x.word, 0) + 1
                    if begin_n >= 3:
                        w3 = line_cuted[i - 3].word
                        if line_cuted[i - 3].flag in replace_type:
                            w3 = line_cuted[i - 3].flag
                        lr_3_dict[w3] = lr_3_dict.get(w3, {})
                        lr_3_dict[w3][w2] = lr_3_dict[w3].get(w2, {})
                        lr_3_dict[w3][w2][w1] = lr_3_dict[w3][w2].get(w1, {})
                        lr_3_dict[w3][w2][w1][x.word] = lr_3_dict[w3][w2][w1].get(x.word, 0) + 1
    all_dicts = {
        'lr_1_dict': lr_1_dict,
        'lr_2_dict': lr_2_dict,
        'lr_3_dict': lr_3_dict,
        'rl_1_dict': rl_1_dict,
        'rl_2_dict': rl_2_dict,
        'rl_3_dict': rl_3_dict
    }
    joblib.dump(all_dicts, join(dict_save_path, 'all_dicts.dict'))


def delect_words():
    # 把每一个候选，通过相互比较编辑距离，以及频率的比值，删选清洗一部分频率和编辑距离过低的候选
    all_dicts = joblib.load(join(dict_save_path, 'all_dicts.dict'))
    lr_1_dict = all_dicts['lr_1_dict']
    lr_2_dict = all_dicts['lr_2_dict']
    lr_3_dict = all_dicts['lr_3_dict']
    rl_1_dict = all_dicts['rl_1_dict']
    rl_2_dict = all_dicts['rl_2_dict']
    rl_3_dict = all_dicts['rl_3_dict']
    freq_multiple = math.exp(1)

    def get_pinyin(word):
        return pinyin.get(word, format="strip", delimiter=" ").split(' ')

    def edit_distance(word1, word2):
        if word1 == word2:
            return 0
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

    def delect_fn(dicts):
        exist_dics = dict()
        kn = 0
        for _k in dicts.keys():
            kn += 1
            # print('\rkn:{} / {}'.format(kn, len(dicts)), end='  ')
            if len(exist_dics) == 0:
                exist_dics[_k] = dicts[_k]
                continue
            if dicts[_k] < 3:
                continue
            need_add = False
            del_keys = list()
            for e_k in exist_dics.keys():
                edit_dis = edit_distance(get_pinyin(_k), get_pinyin(e_k))
                if edit_dis == 0:
                    if dicts[_k] > exist_dics[e_k]:
                        del_keys.append(e_k)
                        need_add = True
                    break
                elif edit_dis == 1:
                    if dicts[_k] / exist_dics[e_k] > freq_multiple:
                        del_keys.append(e_k)
                        need_add = True
                    elif dicts[_k] / exist_dics[e_k] > 1 / freq_multiple:
                        need_add = True
                else:
                    need_add = True
            for d_k in del_keys:
                del exist_dics[d_k]
            if need_add:
                exist_dics[_k] = dicts[_k]
        return exist_dics

    print('\nlr_1_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    print('totle length: {}'.format(len(lr_1_dict)))
    for k in lr_1_dict.keys():
        candicates = lr_1_dict[k]
        before_num += len(candicates)
        candicates = delect_fn(candicates)
        after_num += len(candicates)
        lr_1_dict[k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(lr_1_dict), before_num, after_num), end='   ')
    print('\nrl_1_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    for k in rl_1_dict.keys():
        candicates = rl_1_dict[k]
        before_num += len(candicates)
        candicates = delect_fn(candicates)
        after_num += len(candicates)
        rl_1_dict[k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(rl_1_dict), before_num, after_num), end='   ')
    print('\nlr_2_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    for k2 in lr_2_dict.keys():
        for k in lr_2_dict[k2].keys():
            candicates = lr_2_dict[k2][k]
            before_num += len(candicates)
            candicates = delect_fn(candicates)
            after_num += len(candicates)
            lr_2_dict[k2][k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(lr_1_dict), before_num, after_num), end='   ')
    print('\nrl_2_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    for k2 in rl_2_dict.keys():
        for k in rl_2_dict[k2].keys():
            candicates = rl_2_dict[k2][k]
            before_num += len(candicates)
            candicates = delect_fn(candicates)
            after_num += len(candicates)
            rl_2_dict[k2][k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(rl_1_dict), before_num, after_num), end='   ')
    print('\nlr_3_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    for k3 in lr_3_dict.keys():
        for k2 in lr_3_dict[k3].keys():
            for k in lr_3_dict[k3][k2].keys():
                candicates = lr_3_dict[k3][k2][k]
                before_num += len(candicates)
                candicates = delect_fn(candicates)
                after_num += len(candicates)
                lr_3_dict[k3][k2][k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(lr_1_dict), before_num, after_num), end='   ')
    print('\nrl_3_dict:')
    before_num = 0
    after_num = 0
    process_root_num = 0
    for k3 in rl_3_dict.keys():
        for k2 in rl_3_dict[k3].keys():
            for k in rl_3_dict[k3][k2].keys():
                candicates = rl_3_dict[k3][k2][k]
                before_num += len(candicates)
                candicates = delect_fn(candicates)
                after_num += len(candicates)
                rl_3_dict[k3][k2][k] = candicates
        process_root_num += 1
        print('\rprocess: {} / {} . before: {}  after: {}'.format(
            process_root_num, len(rl_1_dict), before_num, after_num), end='   ')

    all_dicts = {
        'lr_1_dict': lr_1_dict,
        'lr_2_dict': lr_2_dict,
        'lr_3_dict': lr_3_dict,
        'rl_1_dict': rl_1_dict,
        'rl_2_dict': rl_2_dict,
        'rl_3_dict': rl_3_dict
    }
    joblib.dump(all_dicts, join(dict_save_path, 'all_dicts_d2.dict'))


def test2():
    import pinyin
    print(pinyin.get('出了交通事故不赔钱需要坐牢吗', format="strip", delimiter=" "))


if __name__ == '__main__':
    cal_fre_dicts()
    # test2()
    # delect_words()

"""
数量指数测试  e： process: 278 / 363  accuracy:76.58% 
            4： process: 293 / 386  accuracy:75.91%
            10： process: 295 / 390  accuracy:75.64%
            2： process: 352 / 467  accuracy:75.37%3
            无： process: 131 / 217  accuracy:60.37%
"""
