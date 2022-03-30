import re
##  https://tutorials.pytorch.kr/beginner/saving_loading_models.html
import torch
import numpy as np
from transformers import BertTokenizer
from kobert_tokenizer import KoBERTTokenizer
from Bert_Gpt_Chatbot.model.kobert_model import BERTClassifier

ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)

add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tokenizer.add_tokens(add_token)
model = torch.load('../output/model/CSW_KoBert_3multi_0.92.pt', map_location=device)


# model.to(device)


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100


def transform(data):
    # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data, max_length=144, padding="max_length", truncation=True, )
    # data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


# print(tokenizer.tokenize('안녕하세요'))
# exit()

def testModel(model, sentence):
    # cate = ['비욕설', '욕설']
    cate = ['일반_긍정', '섹슈얼', '혐오']

    sentence = transform(sentence)
    input_ids = torch.tensor([sentence[0]]).to(device)
    token_type_ids = torch.tensor([sentence[1]]).to(device)
    attention_mask = torch.tensor([sentence[2]]).to(device)
    print(input_ids)

    result = model(input_ids, token_type_ids, attention_mask)
    idx = result.argmax().cpu().item()
    print("문장에는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result, idx)))



while True:
    s = input('input: ')
    if s == 'quit':
        break
    testModel(model, s)
