import torch
import torch.nn as nn
from transformers import BertModel

### KcBert 모델 설정 클래스
class kcBERTClassifier(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=3, dr_rate=0.2, vocab_size=0, add_token=0):
        super(kcBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('beomi/kcbert-large', return_dict=False)

        if add_token:
            self.bert.resize_token_embeddings(vocab_size + add_token)
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooler = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask, return_dict=False)

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
