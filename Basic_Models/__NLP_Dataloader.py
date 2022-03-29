import os, pickle
import numpy as np
import pandas as pd


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from __Korean_Package import *

### 불용어 제거 함수
def DEF_Stopwords(processed_X, predict=False) :
    stopwords_path = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
                      '도', '를', '으로', '자', '에', '와', '한', '하다']
    if predict != True:
        for i in range(len(processed_X)) :
            words = []
            for j in range(len(processed_X[i])):
                if processed_X[i][j] not in stopwords_path:
                    words.append(processed_X[i][j])
            processed_X[i] = ' '.join(words)
            return processed_X
    else :
        new_text = []
        for i in processed_X:
            if i not in stopwords_path:
                new_text.append(i)
        return new_text


### 카테고리를 원핫엔코딩하는 함수.
def DEF_Onehot(Y, korean_package) :
    encoder = LabelEncoder()
    labeled_Y = encoder.fit_transform(Y)
    label = encoder.classes_
    with open(f'./pickled_ones/{korean_package}_encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)
    onehot_Y = to_categorical(labeled_Y)
    np.save(f'./saved_np/{korean_package}_onehot_data.npy', onehot_Y)
    return onehot_Y


### 토큰화된 문장 데이터의 빈도 조사를 통해 유의미한 빈도로 나타난 단어만을 사용하는 함수.
### 어떤 전처리 패키지를 사용했는지에 따라 토큰화 정보가 다 다르기 때문에 유의미한 단어집합 구성을 위해 함수화하였다.

def DEF_Check_Words_Info(processed_X_with_stopped, threshold=3) :   # 3회 미만의 빈도수
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_X_with_stopped)
    threshold = threshold
    rare_cnt, total_freq, rare_freq = 0, 0, 0
    total_cnt = len(tokenizer.word_index)
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    Vocab_size = total_cnt - rare_cnt + 1

    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    return Vocab_size + 2  # 패딩 토큰과 OVV 토큰을 고려하여 + 2 하자.


### 토큰화된 문장 시퀀스의 길이 정보를 파악해 유의미한 길이만을 반영할수있도록 자동화하는 함수.
### 이상치를 제거하고 99퍼센트의 시퀀스를 포함하도록 설정.

def DEF_Check_Padding_Length(X_train, max_len=10, include=99) :     # include: 99퍼센트의 시퀀스가 포함되도록 설정.
    while True:
        count = 0
        for sentence in X_train :
            if(len(sentence) <= max_len) :
                count += 1
        percent = (count / len(X_train))*100
        if percent > include :
            # print('percent : ', percent, 'max_len : ', max_len, )
            return percent, max_len
            break
        else :
            max_len += 1


### 텐서플로우 tokenize로 정수인코딩함.

def DEF_Tokenizing(processed_X_with_stopped, Vocab_size, korean_package) :

    # 해당 전처리 패키지를 사용한 tokenizer의 유무에 따라 새로 생성할 지, 불러올 지 판단함.
    if f'{korean_package}_tokenizer.pickle' not in os.listdir('./pickled_ones'):
        tokenizer = Tokenizer(Vocab_size, oov_token='OOV')
        tokenizer.fit_on_texts(processed_X_with_stopped)
        with open(f'./pickled_ones/{korean_package}_tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
    else :
        with open(f'./pickled_ones/{korean_package}_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
    tokened_X = tokenizer.texts_to_sequences(processed_X_with_stopped)
    return tokened_X

### 패딩
def DEF_Padding_Sequences(tokened_X, max_len) :
    X_pad = pad_sequences(tokened_X, max_len)
    return X_pad

### 카테고리 비율에 맞춰서 학습-검증 데이터셋 구성
def DEF_Sample_Split(X_pad, Onehot_Y, max_len, vocab_size, test_size, korean_package):
    X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Onehot_Y,
                                                        test_size=test_size, stratify=Onehot_Y)
    if korean_package :
        XY = X_train, X_test, Y_train, Y_test
        np.save(f'./saved_np/{korean_package}_badlang_dataset_{max_len}_{vocab_size}.npy', XY)
    # 재사용을 위해 변수 저장.
    return X_train, X_test, Y_train, Y_test



### csv파일로 저장된 문자 데이터를 불러와 독립변수 / 종속변수로 나눠준다.
def DEF_Setting_Data_to_DF(file_path):

    df = pd.read_csv(file_path, sep='|')
    df.columns = ['비속어','카테고리','text']
#########################################################

    # df['카테고리'].value_counts()
    # longlong = len(df[df['카테고리'] == '혐오'])
    # upsampling_df = pd.DataFrame()
    # test_df = pd.DataFrame()
    # for i in ['일반_긍정','섹슈얼','혐오'] :
    #     test_df = df[df['카테고리'] == '섹슈얼']
    #     test_df = test_df.sample(n=longlong, replace=True, axis=0)
    #     pd.concat(upsampling_df, test_df)
    # upsampling_df.info()

    # test_df = df[df['카테고리'] == '섹슈얼']
    # test_df.info()
    # test_df = test_df.sample(n=2500, replace=True, axis=0)
    # test_df.info()



#########################################################
    df.info()
    # df['text'] = df['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    # print(df['text'].isnull().sum())
    # df['카테고리'].value_counts().plot(kind = 'bar')
    # plt.show()
    X, Y = df['text'], df['카테고리']
    # X, Y = df['text'], df['label']
    return X, Y


### 첫번째 인자로 해당 전처리 패키지 이름을 입력 시  위의 함수들을 이용해서
### 토큰화, 정수인코딩, 시퀀스 길이 조정, 단어집합 크기 조정, 학습-검증 데이터셋 구성까지 자동화한다.

def DEF_Prepare_Dataset(korean_package, eval = True) :
    korean_package = korean_package

    # file_path = './datasets0225/final_train_test_datasets_normalize_0225.csv'
    file_path = 'datasets/data0303/concat_train_test_datasets_0303.csv'

    X, Y = DEF_Setting_Data_to_DF(file_path)
    Onehot_Y = DEF_Onehot(Y, korean_package)

    if korean_package != 'spm':
        if korean_package == 'okt_morphs' :
            processed_X = DEF_Okt_Morphs(X)
            processed_X_with_stopped = DEF_Stopwords(processed_X)   ## 일단 이것만 스탑워드 함수 사용.
        elif korean_package == 'okt_pos' :
            processed_X = DEF_Okt_Pos(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'jamo' :
            processed_X = DEF_Jamotools(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'char' :
            processed_X = DEF_Char(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'mecab' :
            processed_X = DEF_Mecab(X)
            processed_X_with_stopped = processed_X
        elif korean_package == 'soynlp' :
            processed_X = DEF_Soynlp(X)
            processed_X_with_stopped = processed_X

        Vocab_size = DEF_Check_Words_Info(processed_X_with_stopped, 3)
        tokened_X = DEF_Tokenizing(processed_X_with_stopped, Vocab_size, korean_package)
        percent, Max_len = DEF_Check_Padding_Length(tokened_X)

    elif korean_package == 'spm':
        processed_X = DEF_Sentencepiece(X)
        tokened_X = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        Vocab_size = 30000 + 1
        percent, Max_len = DEF_Check_Padding_Length(tokened_X)



    X_pad = DEF_Padding_Sequences(tokened_X, Max_len)
    X_train, X_test, Y_train, Y_test = DEF_Sample_Split(
        X_pad, Onehot_Y, Max_len, Vocab_size, 0.2, korean_package)

    return X_train, X_test, Y_train, Y_test, Max_len, Vocab_size


if __name__ == '__main__':
    # print(DEF_Prepare_Dataset())


    # df = pd.read_csv('./datasets/data0303/concat_datasets_0303.csv', sep='|')
    # X = df['text']
    # X = DEF_Okt_Morphs(X)
    # DEF_Check_Words_Info(X)

    pass

