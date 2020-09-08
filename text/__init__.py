#-*- coding: utf-8 -*-
from text import cleaners
from jamo import h2j
from itertools import chain
from text import cleaners
from text.symbols import symbols
import string
import numpy as np
from text import cleaners
from jamo import h2j
from itertools import chain

import re
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
hangul_symbol =     u'''␀␃%"ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆞᆢᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆪᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀ'''
hangul_symbol_hcj = u'''␀␃%"ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎᅌㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣᆞᆢㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌ'''

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def date_to_hangul(text):
    pattern1 = r'([12]\d{3}.(0[1-9]|1[0-2]).(0[1-9]|[12]\d|3[01]))'
    dates = re.findall(pattern1, text)
    for d in dates:
        date_ = d[0]
        date = date_.replace('/','.').replace('-','.').strip()
        date_digits = date.split('.')
        date = number_to_hangul(date_digits[0]) + '년 ' + number_to_hangul(date_digits[1]) + '월 ' + number_to_hangul(date_digits[2]) + '일'
        text = text.replace(date_, date, 1)

    pattern2 = r'([12]\d{3}.(0[1-9]|1[0-2]))'
    dates = re.findall(pattern2, text)
    for d in dates:
        date_ = d[0]
        date = date_.replace('/','.').replace('-','.').strip()
        date_digits = date.split('.')
        date = number_to_hangul(date_digits[0]) + '년 ' + number_to_hangul(date_digits[1]) + '월'
        text = text.replace(date_, date, 1)
    return text

def number_to_hangul(text):
    temp_text= text
    for idx,char in enumerate(text):
        if (not char in string.digits) and char!=',' and char!='.':
            temp_text=temp_text.replace(temp_text[idx], '_')
    numbers = temp_text.split('_')

    for num in numbers:
        number = num.replace('.', ' ').strip()
        number = number.replace(' ', '.').strip()
        number = number.replace(',', ' ').strip()
        number = number.replace(' ', ',').strip()
        if (number.replace(',','').replace('.','')).isnumeric():
            number_text = digit2txt(number)
            text = text.replace(number, number_text, 1)
    text = text.replace(' ,', ',').replace(',', ', ').replace(' .', '.').replace('.', '. ')
    return text


def digit2txt(strNum):
    # 만 단위 자릿수
    tenThousandPos = 4
    # 억 단위 자릿수
    hundredMillionPos = 9
    txtDigit = ['', '십', '백', '천', '만', '억']
    txtNumber = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
    txtPoint = '쩜 '
    resultStr = ''
    digitCount = 0
    #자릿수 카운트
    for ch in strNum:
        # ',' 무시
        if ch == ',':
            continue
        #소숫점 까지
        elif ch == '.':
            break
        digitCount = digitCount + 1
    digitCount = digitCount-1
    index = 0
    while True:
        notShowDigit = False
        ch = strNum[index]
        #print(str(index) + ' ' + ch + ' ' +str(digitCount))
        # ',' 무시
        if ch == ',':
            index = index + 1
            if index >= len(strNum):
                break
            continue
        if ch == '.':
            resultStr = resultStr + txtPoint
        else:
            #자릿수가 2자리이고 1이면 '일'은 표시 안함.
            # 단 '만' '억'에서는 표시 함
            if(digitCount > 1) and (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos) and int(ch) == 1:
                resultStr = resultStr + ''
            elif int(ch) == 0:
                resultStr = resultStr + ''
                # 단 '만' '억'에서는 표시 함
                if (digitCount != tenThousandPos) and  (digitCount != hundredMillionPos):
                    notShowDigit = True
            else:
                resultStr = resultStr + txtNumber[int(ch)]
        # 1억 이상
        if digitCount > hundredMillionPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-hundredMillionPos]
        # 1만 이상
        elif digitCount > tenThousandPos:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount-tenThousandPos]
        else:
            if not notShowDigit:
                resultStr = resultStr + txtDigit[digitCount]
        if digitCount <= 0:
            digitCount = 0
        else:
            digitCount = digitCount - 1
        index = index + 1
        if index >= len(strNum):
            break
    resultStr = resultStr.replace('일십', '십')
    return resultStr

#
# #
# def get_hangul_to_ids():
#     hangul_to_ids = {char: idx for idx, char in enumerate(hangul_symbol)}
#     ids_to_hangul = {idx: char for idx, char in enumerate(hangul_symbol)}
#     return hangul_to_ids, ids_to_hangul
#
#
def clean_text(txt):
    ### transform english char to korean text
    transform_dict = {'a': '에이', 'b': '비', 'c': '시', 'd': '디', 'e': '이', 'f': '에프', 'g': '지', 'h': '에이치', 'i': '아이',
                      'j': '제이', 'k': '케이', 'l': '엘', 'm': '엠',
                      'n': '엔', 'o': '오', 'p': '피', 'q': '큐', 'r': '아르', 's': '에스', 't': '티', 'u': '유', 'v': '브이',
                      'w': '더블유', 'x': '엑스', 'y': '와이', 'z': '제트',
                      u"'": u'"', '(': ', ', ')': ', ', '#': '샵', '%': '프로', '@': '고팽이', '+': '더하기', '-': '빼기',
                      ':': '나누기', '*': '별'}
    ### remove not allowed chars
    # not_allowed_characters = list('^~‘’')
    # txt = ''.join(i for i in txt if not i in not_allowed_characters)
    txt = txt.lower().strip()
    ### transform special char to hangul
    for k, v in transform_dict.items():
        txt = txt.replace(k, v)
    txt = txt.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace('.', '. ').replace('?', '? ').replace('!', '! ').strip()
    while True:
        if '  ' in txt:
            txt = txt.replace('  ', ' ')
        else:
            break
    return txt


def hangul_to_sequence(hangul_text):
    # load conversion dictionaries
    ### clean number
    hangul_text_ = date_to_hangul(hangul_text)
    hangul_text_ = number_to_hangul(hangul_text_)
    hangul_text_ = clean_text(hangul_text_)
    ### add end of sentence symbol
    hangul_text_ = hangul_text_ + u"␃"  # ␃: EOS
    ### get dictionary of chars
    hangul_to_ids= _symbol_to_id
    ### process jamos
    text = [h2j(char) for char in hangul_text_]
    text = chain.from_iterable(text)
    hangul_text_ = [h2j(char) for char in text]
    hangul_text_ = chain.from_iterable(hangul_text_)
    sequence = []
    try:
        ### convert jamos to ids using dictionary
        for char in hangul_text_:
            if char in symbols:
                sequence.append(hangul_to_ids[char])
            else:
                try:
                    print(char)
                    sequence.append(hangul_to_ids[symbols[hangul_symbol_hcj.index(char)]])
                except Exception as e:
                    sequence.append(hangul_to_ids['.'])
    except KeyError as e:
        raise KeyError('KeyError (at key: {}) when processing: {}'.format(e,hangul_text))
    return sequence

def text_to_sequence_(text, cleaner_names):
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '_' and s is not '~'


def group_words(s):
    return re.findall(u'[a-z]+', s)

def text_to_sequence(txt, cleaner, lang=0):
    txt_=txt.lower().strip()
    sequence=[]
    if lang==0:
        sequence += hangul_to_sequence(txt_)[:-1]
    else:
        sequence += text_to_sequence_(txt_, cleaner)
    mask = (np.array(sequence)<94).astype(int)
    sequence += [1]
    return sequence, mask


# print(_symbol_to_id)
#Unfortunately he took to drink
#txt='안녕하~[세요]'
# txt = 'and a heavy fee at the ~ rate of 8 안녕 per 100, with 4 for 여러분 every [additional] hundred.'

#print(text_to_sequence(txt, cleaner=['english_cleaners'], lang=0)[0])
# sequence_to_text(text_to_sequence(txt, ['english_cleaners'], lang=0)[0])