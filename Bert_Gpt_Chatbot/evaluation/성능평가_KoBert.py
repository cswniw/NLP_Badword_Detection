import pickle, torch
import numpy as np
import pandas as pd
import torch
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics import *


ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
tokenizer.add_tokens(add_token)
model = torch.load('../output/model/CSW_KoBert_3multi_0.92.pt', map_location=device)
model.to(device)

### 모델 예측값 출력
def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100

### 전처리 함수
def transform(data):
    # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    # data = tokenizer(data)
    data = tokenizer(data, max_length=144, padding="max_length", truncation=True, )
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

### 예측값 출력 함수
def testModel(model, sentence):
    cate = ['일반_긍정', '섹슈얼', '혐오']
    sentence = transform(sentence)

    input_ids = torch.tensor([sentence[0]]).to(device)
    token_type_ids = torch.tensor([sentence[1]]).to(device)
    attention_mask = torch.tensor([sentence[2]]).to(device)

    result = model(input_ids, token_type_ids, attention_mask)
    idx = result.argmax().cpu().item()

    return cate[idx]


### 평가 데이터셋 불러오기
df = pd.read_csv('../input/datasets0225/final_evaluation_datasets_normalize_0225.csv', sep="|")
X = df['text']
Y = df['카테고리']
Z = []

###  첫 실행시만 주석 해제
for i in range(len(X)):
    Z.append(testModel(model, X[i]))
    if i % 100 == 0:
        print("★", end=" ")
print()

with open(f'./pickled_ones/KoBERT_Z_test.pickle', 'wb') as f:
    pickle.dump(Z, f)

### 이전 파일 읽어올때 주석 해제
# with open(f'./pickled_ones/KoBERT_Z_test.pickle', 'rb') as f:
#     Z = pickle.load(f)

### 정답,예측 비교
df_a = pd.DataFrame({"text": X, "정답": Y, "예측": Z})
# print(df_a)
df_a.to_csv(f'./output/정답비교_KoBERT.csv', index=False)


### 예측 잘못된 데이터들 확인
target_names = ['섹슈얼', '일반_긍정', '혐오']

for a1 in target_names:
    for a2 in target_names:
        if a1 != a2:
            error_df = []
            for i in range(len(X)):
                if Y[i] == a1 and Z[i] == a2:
                    error_df.append(X[i])
            error_df = pd.Series(error_df, name=f"{a1}->{a2}")
            # print(error_df)
            error_df.to_csv(f'./output/KoBERT_{a1}_{a2}_error_data.csv', index=False)

### 성능평가 데이터
print("==================KoBERT==================")
print(confusion_matrix(Y, Z))
print(classification_report(Y, Z, target_names=target_names))