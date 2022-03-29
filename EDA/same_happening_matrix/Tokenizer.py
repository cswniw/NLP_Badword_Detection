import re
import pickle
import pandas as pd
import emoji
from transformers import AutoTokenizer
from soynlp.normalizer import repeat_normalize
from konlpy.tag import Okt


# okt 토큰화
def okt_tokenize(corpus):
    okt = Okt()
    all_tokens = []
    for t in corpus['text']:
        data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", t)
        data = okt.pos(data, stem=True, norm=True)
        tokens = []
        for token in data:
            if len(token[0]) > 1:
                tokens.append(token)
        all_tokens.append(tokens)

    tokens_text = []
    for tokens in all_tokens:
        temp = []
        for token in tokens:
            if (token[1] == 'Noun'):
                #  | (token[1] == 'Verb') | (token[1] == 'Adjective')
                temp.append(token[0])
        tokens_text.append(temp)

    # 토큰화 문장 이어붙이기
    tokened_text = []
    for tokens in tokens_text:
        tokened_text.append(' '.join(tokens))

    # 토큰화 문장 저장
    df_tokened_text = pd.DataFrame({'비속어': corpus['비속어'], '카테고리': corpus['카테고리'], 'text': tokened_text})
    df_tokened_text.to_csv('tokened.csv', index=False)

    return all_tokens, tokens_text, tokened_text


# wordpiece 토큰화
def wordpiece_tokenize(corpus):
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    def clean(x):
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        x = tokenizer.tokenize(x)
        return x

    all_tokens = []
    tokens_text = []
    for text in corpus['text']:
        temp = clean(text)
        tokens = []
        for token in temp:
            if ('#' not in token and len(token) > 1) or (len(token.replace('#', '')) > 2):
                tokens.append(token.replace('#', ''))
        all_tokens.append(temp)
        tokens_text.append(tokens)

    # 토큰화 문장 이어붙이기
    tokened_text = []
    for tokens in tokens_text:
        tokened_text.append(' '.join(tokens))

    return all_tokens, tokens_text, tokened_text
