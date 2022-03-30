import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


### KoGPT 모델 설정 클래스
class GPT2Chat(nn.Module):
    def __init__(self):
        super(GPT2Chat, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    def generate(self, input_ids,
                 do_sample=True,
                 max_length=0,
                 top_p=0.92,    # 상위 92프로 확률 단어 선택
                 top_k=50,      # 확률 순 상위 50개 단어 선택
                 temperature=0.6,
                 no_repeat_ngram_size=None,
                 num_return_sequences=3,    # 단어 반복 3번까지 허용
                 early_stopping=False,):
        return self.gpt2.generate(input_ids,
                                  do_sample=do_sample,
                                  max_length=max_length,
                                  top_p=top_p,
                                  top_k=top_k,
                                  temperature=temperature,
                                  no_repeat_ngram_size=no_repeat_ngram_size,
                                  num_return_sequences=num_return_sequences,
                                  early_stopping=early_stopping,)

    def forward(self, input, labels=None):
        if labels is not None:
            outputs = self.gpt2(input, labels=labels)
        else:
            outputs = self.gpt2(input)

        return outputs
