import numpy as np
from torch.utils.data import Dataset
from kobert_tokenizer import KoBERTTokenizer
from transformers import PreTrainedTokenizerFast, BertTokenizer

### KoBert, KcBert, KoGPT 모델 사용 시 해당 모델의 데이터로더 클래스를 불러온다.

### Kobert 모델 데이터로더
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx=0, label_idx=1, max_len=144, add_token=0):
        self.dataset = dataset
        self.max_len = max_len    # 입력 시퀀스 최적 길이 144
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.vocab_size = self.tokenizer.vocab_size
        if add_token:   # 토큰이 추가로 있는 경우
            self.added_token_num = self.tokenizer.add_tokens(add_token)

        self.sentences = [self.transform(i[sent_idx]) for i in self.dataset]
        # 아래의 transform 함수를 사용하여 Bert모델에 맞게 전처리해준다.
        self.labels = [np.int32(i[label_idx]) for i in self.dataset]
        # 레이블은 넘피형식으로 바꿔줌.

    def transform(self, data):
        # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
        # 정규표현식이 필요하다면 사용. 비속어 데이터의 경우 검열을 피하기 위한 용도로 기호가 자주 쓰이므로 사용하지 않았다.
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)


### KoGPT 모델 데이터로더
class GPT2Dataset(Dataset):
    def __init__(self, dataset, max_len=70):
        self.dataset = dataset
        self.max_len = max_len  # 대화 데이터셋 최적 길이 70
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                    bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',)
        self.q_token = "<usr>"      # 질문 문장 시작 토큰
        self.a_token = "<sys>"      # 답변 문장 시작 토큰
        self.mask_token = '<unused0>'   # 마스크 토큰
        self.sent_token = '<unused1>'   # 질문 문장 종료 토큰

        ######
        # self.categories = []
        ######

    def transform(self, data):
        question = data["Q"]
        # question = re.sub(r"([?.!,])", r" ", question)    ## 정규표현식 사용 시
        # sentiment = str(data['label'])        ## 질문 문장의 레이블 정보가 필요시 추가

        q_toked = self.tokenizer.tokenize(self.q_token + question + self.sent_token)
        q_len = len(q_toked)

        answer = data["A"]
        # answer = re.sub(r"([?.!,])", r" ", answer)
        a_toked = self.tokenizer.tokenize(self.a_token + answer + self.tokenizer.eos_token)
        a_len = len(a_toked)


        ### 질문+답변 문장 데이터가 최대 시퀀스 길이를 넘어가지 않도록 조정한다.
        if q_len + a_len > self.max_len:
            q_len = self.max_len - 35
            a_len = self.max_len - q_len

            # 길이 조정하더라도 스페셜토큰을 자르면 안되므로 아래와 같이 조정한다.
            q_toked = q_toked[:q_len - 1] + q_toked[-1:]
            a_toked = a_toked[:a_len - 1] + a_toked[-1:]


        ### token_ids
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대 시퀀스 길이에 못 미치면 남는 부분은 패딩토큰으로 채움.
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        ### mask   질문 문장과 답변 문장을 구분
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        ### labels
        labels = [self.mask_token, ] * q_len + a_toked[1:]
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        # return token_ids, np.array(mask), labels_ids

        return token_ids, np.array(mask), labels_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        return self.transform(data)



### KcBert 모델 데이터로더 (KoBert와 유사)
class kcBERTDataset(Dataset):
    def __init__(self, dataset, sent_idx=0, label_idx=1, max_len=144, add_token=0):
        self.dataset = dataset
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')
        self.vocab_size = self.tokenizer.vocab_size
        if add_token:
            self.added_token_num = self.tokenizer.add_tokens(add_token)


        self.sentences = [self.transform(i[sent_idx]) for i in self.dataset]
        self.labels = [np.int32(i[label_idx]) for i in self.dataset]

    def transform(self, data):
        # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)   # 원본
        data = self.tokenizer(data, max_length=self.max_len, padding="max_length", truncation=True,)
        return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx] + (self.labels[idx],)