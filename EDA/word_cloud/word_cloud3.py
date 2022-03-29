import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from pylab import rcParams
from wordcloud import WordCloud, STOPWORDS
from matplotlib import font_manager, rc
from Basic_models.__Korean_Package import *
from soynlp.normalizer import *
from openpyxl import load_workbook, Workbook
import copy

font_path = "NanumGothicBold.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def show_wordcloud(data, stopwords):
    wordcloud = WordCloud(font_path = font_path,
    background_color='white', max_words=200, max_font_size=100,stopwords=stopwords, scale=3, random_state=2)
    wordcloud=wordcloud.generate(str(data))
    fig = plt.figure(2, figsize=(12, 12))
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

def Double_wordcloud(data, data2, stopwords, pos, category):
    white_wordcloud = WordCloud(font_path=font_path,
                                background_color='white', max_words=200, max_font_size=100, stopwords=stopwords,
                                scale=3, random_state=2).generate(str(data))

    black_wordcloud = WordCloud(font_path=font_path,
                                background_color='black', max_words=200, max_font_size=100, stopwords=stopwords,
                                scale=3,
                                random_state=2).generate(str(data2))
    plt.rcParams['font.family'] = 'NanumGothic'
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    fig.tight_layout()
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(white_wordcloud)
    ax[1].imshow(black_wordcloud)
    plt.savefig('./wordcloud_img/wordclud_{}_{}.png'.format(pos, category))
    plt.show()

def DEF_Okt_Pos_v3(Y, pos ,predict=False):
    okt = Okt()
    X = copy.deepcopy(Y)
    for i in range(len(X)):
        X[i] = okt.pos(X[i])
        words = []

        for j in X[i]:
            if j[1] == pos:
                words.append(j[0])
            X[i] = words

    processed_Y = X

    return processed_Y



if __name__ == '__main__':

    category=['혐오','섹슈얼']
    category_eng=['disgust', 'sexsual']
    for j in range(len(category)):
        df = pd.read_csv("../datasets/final_train_test_datasets_normalize_0225.csv", sep='|', encoding='utf-8', header=0)
        # df = pd.read_csv("../input/Feb_21_bad_lang_ready_to_trainset.csv", sep='|', encoding='utf-8', header=0)

        Af = df['text']
        af_swear = df[df['카테고리'] == category[j]] # 1이면 비속어
        af_normal = df[df['카테고리'] == '일반_긍정']
        # af_normal = df[df['카테고리'] == '일상']

        X, Y = af_swear['text'], af_swear['카테고리']
        X = list(af_swear['text'])

        Z= af_normal['text']


        processed_X=[]
        processed_af_normal= []
        for i in X:
            i=emoticon_normalize(str(i))
            # i = DEF_Spacing_Pyko(i, predict=True)
            sentence = re.sub('[^가-힣 ]', '', i)
            processed_X.append(sentence)

        for i in Z:
            i=emoticon_normalize(str(i))
            # i = DEF_Spacing_Pyko(i, predict=True)
            sentence = re.sub('[^가-힣 ]', '', i)
            processed_af_normal.append(sentence)
        # print('c')
        # koreanpakage_list = [DEF_Hanspell, DEF_Mecab, DEF_Soynlp, DEF_Okt_Morphs, DEF_Okt_Pos]
        # pakage= koreanpakage_list[4]


        processed_noun = DEF_Okt_Pos_v3(processed_X, pos='Noun', predict=False)
        processed_verb = DEF_Okt_Pos_v3(processed_X, pos='Verb', predict=False)
        processed_adjective = DEF_Okt_Pos_v3(processed_X, pos='Adjective', predict=False)

        processed_af_normal_noun = DEF_Okt_Pos_v3(processed_af_normal,pos='Noun', predict=False)
        processed_af_normal_verb = DEF_Okt_Pos_v3(processed_af_normal,pos='Verb', predict=False)
        processed_af_normal_adjective = DEF_Okt_Pos_v3(processed_af_normal,pos='Adjective', predict=False)

       # stopwords 추가하기
        stopwords = set()
        stopwords.update([])

        Double_wordcloud(processed_noun, processed_af_normal_noun, stopwords=stopwords,pos='Noun', category= category_eng[j])
        Double_wordcloud(processed_verb, processed_af_normal_verb, stopwords=stopwords, pos='Verb',category= category_eng[j])
        Double_wordcloud(processed_adjective, processed_af_normal_adjective, stopwords=stopwords, pos='Adjective', category= category_eng[j])
