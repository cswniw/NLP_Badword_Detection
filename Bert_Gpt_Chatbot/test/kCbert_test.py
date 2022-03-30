import torch
import numpy as np
from transformers import BertTokenizer

### torch cuda 설정
ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)

### 토큰 추가
add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']

tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
tokenizer.add_tokens(add_token)

### 모델 불러오기
model = torch.load('../output/model/kCbert_3multi_0.96_Mar03.pt', map_location=device)


### 모델이 예측한 확률값으로 문자 데이터가 어떤 카테고리로 분류되고 정확도는 몇일지 출력하는 함수
def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    print('valscpu', valscpu)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    print(np.exp(valscpu[0]))
    print(np.exp(valscpu[1]))
    print(np.exp(valscpu[2]))

    return ((np.exp(valscpu[idx])) / a).item() * 100


### 모델에 입력하기 위해 문자 데이터를 전처리 해준다.
def transform(data):
    # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)  # 원본
    data = tokenizer(data, max_length=144, padding="max_length", truncation=True, )
    # data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


### 입력 데이터를 전처리하여 모델이 예측한 값을 알아보기 쉽게 해석하는 함수.
def testModel(model, sentence):
    # cate = ['비욕설', '욕설']
    cate = ['일반_긍정', '섹슈얼', '혐오']

    sentence = transform(sentence)  # 위의 transform 함수 호출.
    input_ids = torch.tensor([sentence[0]]).to(device)      # 토치 텐서 형식으로 바꿈
    token_type_ids = torch.tensor([sentence[1]]).to(device)     # 토치 텐서 형식으로 바꿈
    attention_mask = torch.tensor([sentence[2]]).to(device)     # 토치 텐서 형식으로 바꿈
    # print(input_ids)

    result = model(input_ids, token_type_ids, attention_mask)

    print(result)
    idx = result.argmax().cpu().item()
    print("문장에는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result, idx)))


while True:
    s = input('input: ')
    if s == 'quit':
        break
    testModel(model, s)

