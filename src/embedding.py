# -*- coding: utf-8 -*-
'''
전처리된 댓글들을 임베딩한다.
'''
import numpy as np
import joblib


def char2vec(char):
    # (사전 예측된 dict를 통한) 글자 임베딩 수행
    if char == '~':
        # 빈 데이터 -> [0., 0., .., 0.]
        return np.array([0.] * len(vecdict['ㄱ']))
    return vecdict[char]


def embedding(x):
    # 데이터에 대해 임베딩을 수행
    return np.array([[char2vec(e) for e in ele] for ele in x])


def padding(x, length=256, pad=None):
    # 패딩을 수행
    result = []
    for n, ele in enumerate(x):
        if len(ele) == length:
            result.append(ele)
            continue
        if pad is None:
            pad = [0.] * len(ele[0])

        a, b = np.array(ele), np.array([pad] * (length - len(ele)))
        try:
            mid = np.concatenate((a, b))
        except:
            continue

        result.append(mid)
    return np.array(result)


def padding_x(x, length=256, pad=None):
    # 하나의 input 값에만 padding 수행
    if len(x) > length:
        return None
    if len(x) == length:
        return x
    if pad == None:
        pad = [0.] * len(x[0])
    a, b = np.array(x), np.array([pad] * (length - len(x)))
    try:
        mid = np.concatenate((a, b))
    except:
        None
    return mid


vecdict = joblib.load('models/char2vec.dic')  # 각 글자에 대응하는 vector가 담긴 dictionary
