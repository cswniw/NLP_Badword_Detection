import numpy as np
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer


# 동시 발생 행렬
def Co_Matrix(corpus, vocab_size, window_size=1):
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for c in corpus:
        for idx, word_id in enumerate(c):
            text_size = len(c)
            for i in range(1, window_size+1):
                left_idx = idx - i
                right_idx = idx + i
                if left_idx >= 0:
                    left_word_id = c[left_idx]
                    co_matrix[word_id, left_word_id] += 1
                if right_idx < text_size:
                    right_word_id = c[right_idx]
                    co_matrix[word_id, right_word_id] += 1
    return co_matrix


# PPMI
def PPMI(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float)  # C와 동일한 형상을 가진 zeros행렬
    N = np.sum(C)  # C 원소의 총 합 = 말뭉치에 있는 총 단어의 수에 비례하는 값.
    S = np.sum(C, axis=0)  # axis = 0이므로 그 단어가 총 몇번 나왔는지에 비례하는 값.
    total = C.shape[0] * C.shape[1]  # 중간 결과 출력 위해.
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)
            if verbose:
                cnt += 1
                if cnt % (total // 10) == 0:
                    print('%.1f%% 완료' % (100 * cnt / total))
    return M


# PPMI기반 유사 단어 출력
def most_similar(word, word_key, word_value, word_matrix, top=5):
    if word not in word_key:
        print('%s(을)를 찾을 수 없습니다.' % word)
        return

    print('\n[word] ' + word)
    query_id = word_value[word_key.index(word)]
    print(query_id)
    query_vec = word_matrix[query_id]

    sim_scores = list(enumerate(query_vec))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top+1]

    for score in sim_scores:
        print(word_key[score[0]], score[1])


# word embedding용 word to num
def tokens_to_num(tokens_text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens_text)
    tokenized = tokenizer.texts_to_sequences(tokens_text)
    with open('token.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)

    word_dic = tokenizer.word_index
    tokens_key = list(word_dic.keys())
    tokens_value = [value - 1 for value in list(word_dic.values())]
    vocab_size = len(tokens_value)
    print(vocab_size, tokens_key[:5], tokens_value[:5])

    # 토큰화는 1부터 시작 -> 0으로
    tokenized_minus = []
    for tokens in tokenized:
        token_minus =[]
        for token in tokens:
            token_minus.append(token - 1)
        tokenized_minus.append(token_minus)
    tokenized = tokenized_minus
    return tokenized, tokens_key, tokens_value, vocab_size


# TF-IDF
# min_df : 특정 단어가 등장하는 최소 문장의 수
# analyzer : 학습 단위(word, char)
# sublinear_tf : TF 값의 스무딩 처리, 데이터가 클수록 효과적
# ngram_range : 단어 묶음 개수 ex) ngram_range = (1, 2) 이면 vocab에 '밥 먹자' 등 두개 단위의 단어또한 포함
# max_features : 단어의 개수 제한
def TFIDF(corpus, min_df):
    tfidfv = TfidfVectorizer(min_df=min_df, sublinear_tf=True)
    tfidfv.fit(corpus)
    tfidfv_words = sorted(tfidfv.vocabulary_.items())
    tfidf_matrix = tfidfv.transform(corpus).toarray()
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    with open('tfidf.pickle', 'wb') as f:
        pickle.dump(tfidfv, f)
    return tfidfv_words, tfidf_matrix, cosine_sim


# TF-IDF 기반 특정 문장과 유사한 문장 출력
def TFIDF_sim(cosine_sim, idx):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    sim_texts = [idx[0] for idx in sim_scores]
    return sim_texts



def word2vec_visual(key_word, model):
    sim_word = model.wv.most_similar(key_word, topn=10)

    vectors = []
    labels = []
    for label, _ in sim_word:
        labels.append(label)
        vectors.append(model.wv[label])
    df_vectors = pd.DataFrame(vectors)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_value = tsne_model.fit_transform(df_vectors)
    df_xy = pd.DataFrame({'words': labels, 'x': new_value[:, 0], 'y': new_value[:, 1]})
    df_xy.loc[df_xy.shape[0]] = (key_word, 0, 0)

    plt.figure(figsize=(8, 8))
    plt.scatter(0, 0, s=1500, marker='*')
    for i in range(len(df_xy.x)):
        a = df_xy.loc[[i, (len(df_xy.x) - 1)], :]
        plt.plot(a.x, a.y, '-D', linewidth=2)
        plt.annotate(
            df_xy.words[i], xytext=(5, 2), xy=(df_xy.x[i], df_xy.y[i]), textcoords='offset points', ha='right',
            va='bottom')
    plt.show()