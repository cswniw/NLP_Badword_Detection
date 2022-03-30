import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

from Bert_Gpt_Chatbot.dataloader.dataloader import GPT2Dataset
from Bert_Gpt_Chatbot.model.kogpt2_model import GPT2Chat

### torch cuda 설정
ctx = "cuda" if torch.cuda.is_available() else "cpu"
print(ctx)
device = torch.device(ctx)

### 대화 데이터셋 불러오기
chatbot_file = pd.read_csv('../input/datasets0225/KoGPT_chatbot_10000.csv')
chatbot_file.dropna(inplace=True)

### 하이퍼 파라미터 설정
epochs = 15
batch_size = 8
Sneg = -1e18
learning_rate = 3e-5

### 모델 불러오기
model = GPT2Chat()
model.to(device)

### GPT모델 학습에 적합하게 데이터셋 전처리.
train_dataset = GPT2Dataset(chatbot_file)

### 롱텐서 형식으로 형식 변환
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

### 데이터 로더
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_batch,)

### 가중치, 편향
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
### 옵티마이저 아담
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
### 비용함수
criterion = torch.nn.CrossEntropyLoss(reduction="none")

### loss값으로 최적의 모델을 찾기위해 최소값을 100으로 설정.. 학습마다 갱신됨.
minimum_loss = 100

### 모델 학습 시작
model.train()
for epoch in range(epochs):
    for batch_idx, (token_ids, mask, label) in enumerate(tqdm(train_dataloader)):
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model(token_ids).logits
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss_2 = avg_loss
        avg_loss.backward()
        # 학습 끝
        optimizer.step()

    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    # torch.save(state, '../output/model/checkpoint_chatbot.pt')

    print('epoch:', epoch, 'loss:', avg_loss)

    ### 모델 성능이 좋으면 저장.
    if avg_loss_2 < minimum_loss :

        model.eval()
        # torch.save(model, '../output/model/KoGPT_chatbot_{}.pt'.format(avg_loss_2))
        minimum_loss = avg_loss_2