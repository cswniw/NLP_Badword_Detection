import torch
from transformers import PreTrainedTokenizerFast

### torch cuda 설정
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

## 모델 불러오기
model = torch.load('../output/model/Mar06_KoGPT_chatbot_28.pt', map_location=device)
model.to(device)

### 토크나이저 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',)

q_token = "<usr>"
a_token = "<sys>"
sent_token = '<unused1>'
sent= '0'


with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":     # quit 입력 시 채팅 종료.
            break
        a = ""

        # 입력된 문자 데이터를 스페셜토큰과 함께 엔코딩 후 롱텐서 형식으로 변환
        while 1:
            input_ids = torch.LongTensor(
                tokenizer.encode(q_token + q + sent_token + a_token + a)).unsqueeze(dim=0)
            input_ids = input_ids.to(device)
            # if ctx == 'cuda:0':
            #     input_ids = input_ids.to(ctx)
            pred = model(input_ids)
            pred = pred.logits
            # if ctx == 'cuda:0':
            #     pred = pred.cpu()
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
            if gen == tokenizer.eos_token:  # while문으로 문자 생성 반복 중  문장종료토큰(eos토큰)이 나오면 break.
                break
            a += gen.replace("▁", " ")  # 서브워드 토크나이저의 토큰화 과정에서 나오는 _ 기호를 삭제.
        print("Chatbot > {}".format(a.strip()))