# -*- coding: utf-8 -*-
'''
댓글 데이터를 전처리한다.
Embedding이 아니라 긴 댓글은 제거하거나 특수문자를 제거한다.
'''
import re
import itertools

BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']
JONGSUNG_LIST = ['~', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
GYUP2CHO = {'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ',
            'ㅄ': 'ㅂㅅ'}  # 겹자음을 자음으로 변환



def remain_char(x):
    # 오직 한글 글자만 남기기 (띄어쓰기, 숫자, 특수문자, 영어 등은 삭제)
    return [''.join(re.findall(r'[ㄱ-ㅎㅏ-ㅣ각-힣]', i)) for i in x]  # 숫자도 삭제 (숫자 보존하려면 표현식 뒤에 '0-9' 추가)


def long2short(x):
    # 연속적으로 긴 단어는 간추리기
    # ef) f('ㅋㅋㅋㅋㅋㅋㅋ앜ㅋㅋㅋ') -> f('ㅋ앜ㅋ')
    result = []
    keep = True
    for ele in x:
        while True:
            candidates = set(re.findall(r'(\w)\1', ele))
            repeats = itertools.chain(*[re.findall(r"({0}{0}+)".format(c), ele) for c in candidates])

            keep = False
            for org in [i for i in repeats if len(i) >= 2]:
                ele = ele.replace(org, org[0])
                keep = True
            if not keep:
                break
        result.append(ele)
    return result


def analchar(test_keyword):
    # 글자 -> 초성, 중성, 종성 분리 (한글 아니면 그대로 반환)
    # ex) f('아녕ㅕㄴ') -> 'ㅇㅏ~ㄴㅕㅇㅕ~~ㄴ~~'
    split_keyword_list = list(test_keyword)

    result = []
    for keyword in split_keyword_list:
        # 한글 여부 확인 후 초성, 중성, 종성 분리
        if re.match(r'.*[가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))  # 종성 없으면 char3 = 0 = '~'
            result.append(JONGSUNG_LIST[char3])
        elif re.match(r'[ㄱ-ㅎ]', keyword) is not None:
            result.append(keyword + '~~')
        elif re.match(r'[ㅏ-ㅣ]', keyword) is not None:
            result.append('~' + keyword + '~')
        else:
            result.append(keyword)

    return ''.join(result)


def data2anal(x):
    # 글자 -> 초성, 중성, 종성 분리 (한글 아니면 그대로 반환)
    return [analchar(i) for i in x]


def replace_gyup(x):
    # 겹자음을 자음으로 변환한다.
    # ex) 'ㅄ새끼' -> 'ㅂㅅ새끼'
    result = []
    for ele in x:
        for gyup, cho in GYUP2CHO.items():
            ele = ele.replace(gyup, cho)
        result.append(ele)
    return result


def preprocess(texts):
    texts = remain_char(texts)  # 특수문자, 영어 등 제거
    texts = long2short(texts)   # 연속적인 글자 단축 (ㅋㅋㅋㅋ->ㅋ)
    texts = data2anal(texts)    # 초성, 중성, 종성 분리
    return texts
