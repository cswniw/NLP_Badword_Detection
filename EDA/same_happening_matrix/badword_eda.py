import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud, STOPWORDS
from gensim.models import Word2Vec

from EDA.same_happening_matrix import Tokenizer, WordEmbedding

# 폰트 설정
font_path = "C:/Windows/Fonts/gulim.ttc"
font = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font)


# 데이터 로드 - csv
df = pd.read_csv('../datasets/data0303/concat_datasets_0303.csv', sep='|')
print(df.head())

# 문장 길이 분석
df_len = df['text'].str.len()
print(pd.DataFrame({'text_len': df_len}).describe())

ax = df_len.hist(bins=200, range=(0, 200))
ax.set_xlabel('text')
ax.set_ylabel('count')
plt.show()

# 문장 길이 분석 boxplot
plt.figure(figsize=(8, 9))
plt.boxplot(df_len, labels=['text'])
plt.show()

# 라벨 분포
labels = df['카테고리'].value_counts()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(df['카테고리'])
plt.show()

# # wordpiece
# print('Wordpiece tokenize start')
# all_tokens, tokens_text, tokened_text = Tokenizer.wordpiece_tokenize(df)

# Okt
print('Okt tokenize start')
all_tokens, tokens_text, tokened_text = Tokenizer.okt_tokenize(df)  # Tokenizer.py의 okt_tokenize 함수 호출.


# wordcloud(okt)
# plt.figure(figsize=(15, 10))
# sub = 1
# category = ['일반', '섹슈얼', '혐오']
# wordclass = ['Noun', 'Verb', 'Adjective']
# wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=600, font_path=font_path)
# for wc in wordclass:
#     temp_text = []
#     for tokens in all_tokens:
#         temp_word = []
#         for token in tokens:
#             if token[1] == wc:
#                 temp_word.append(token[0])
#         temp_text.append(' '.join(temp_word))
#     df_temp = pd.DataFrame({'카테고리': df['카테고리'], 'text': temp_text})
#     for cate in category:
#         cloud = wordcloud.generate(' '.join(df_temp[df_temp['카테고리'] == cate]['text']))
#         plt.subplot(len(wordclass), len(category), sub)
#         plt.imshow(cloud)
#         plt.gca().set_title('{}_{}'.format(cate, wc))
#         plt.axis('off')
#         sub += 1
# plt.show()


# tokenize (토큰 -> 숫자)
tokenized, tokens_key, tokens_value, vocab_size = WordEmbedding.\
    tokens_to_num(tokens_text)      # WordEmbedding.py의 tokens_to_num 함수 호출.

# co-occur
window_size = 2
wordvec_size = 100

print("동시 발생 행렬 계산")
C = WordEmbedding.Co_Matrix(tokenized, vocab_size, window_size=window_size)    # WordEmbedding.py의 Co_Matrix 함수 호출.


# print("PPMI 계산")
# W = WordEmbedding.PPMI(C, verbose=True)    # WordEmbedding.py의 PPMI 함수 호출.
#
# print('SVD 계산')
# U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
# svd = U[:, :wordvec_size]

querys = ['식사', '점심', '저녁', '아침', '새끼','고민','병신','세끼']
for query in querys:
    WordEmbedding.most_similar(query, tokens_key, tokens_value, C)      # WordEmbedding.py의 most_similar 함수 호출.


# WordEmbedding
# TF-IDF
print('TF-IDF start')
tfidfv_words, tfidf_matrix, cosine_sim = WordEmbedding.TFIDF(tokened_text, min_df=2)    # WordEmbedding.py의 TFIDF 함수 호출.
print(len(tfidfv_words), cosine_sim.shape)

test_idx = 100
sim_texts = WordEmbedding.TFIDF_sim(cosine_sim, test_idx)   # WordEmbedding.py의 TFIDF_sim 함수 호출.
print(df['text'][test_idx])
print(df['text'].iloc[sim_texts])


# Word2Vec
print('Word2Vec start')
word2vec = Word2Vec(sentences=tokens_text, vector_size=100, window=2, min_count=2, workers=4, sg=0)
print(word2vec.wv.vectors.shape)

for query in querys:
    print(query)
    print(word2vec.wv.most_similar(query))      # WordEmbedding.py의 most_similar 함수 호출.
    WordEmbedding.word2vec_visual(key_word=query, model=word2vec)       # WordEmbedding.py의 word2vec_visual 함수 호출.