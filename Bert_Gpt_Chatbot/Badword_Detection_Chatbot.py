import telegram, random
# pip install python-telegram-bot
# https://python.bakyeono.net/chapter-12-2.html

import torch, re
import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer

### torch cuda 설정
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

### 모델 불러오기
badword_model = torch.load('./output/model/CSW_kCbert_3multi_0.93.pt', map_location=device)
badword_model.to(device)

### 토크나이저 불러오기 및 세팅
# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')

add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
tokenizer.add_tokens(add_token)


### 전처리 함수
def transform(data):
    # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data, max_length=144, padding="max_length", truncation=True, )
    # data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])

### 대화용 챗봇 GPT 모델 불러오기
chat_model = torch.load('./output/model/Mar06_KoGPT_chatbot_28.pt', map_location=device)
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')

### 이하 텔레그램 어플리케이션 연동   비속어 검출 챗봇 구동을 위한 코드
import json
import time  # 추가함
import urllib.parse
import urllib.request


TOKEN = '발급받은 텔레그램 챗봇 토큰'


# 지정한 url의 웹 문서를 요청하여, 본문을 반환한다.
def request(url):
    response = urllib.request.urlopen(url)
    byte_data = response.read()
    text_data = byte_data.decode()
    return text_data


# 텔레그램 챗봇 웹 API에 요청을 보내기 위한 URL을 만들어 반환한다.
def build_url(method, query):
    return f'https://api.telegram.org/bot{TOKEN}/{method}?{query}'


# 텔레그램 챗봇 웹 API에 요청하고 응답 결과를 사전 객체로 해석해 반환한다.
def request_to_chatbot_api(method, query):
    url = build_url(method, query)
    response = request(url)
    return json.loads(response)


# 텔레그램 챗봇 API의 getUpdate 메서드 요청 결과에서 필요한 정보만 남긴다.
def simplify_messages(response):
    result = response['result']
    if not result:
        return None, []
    last_update_id = max(item['update_id'] for item in result)

    try:
        messages = [item['message'] for item in result]
        simplified_messages = [{'from_id': message['from']['id'],
                                'text': message['text']}
                               for message in messages]
    except:
        for message in messages:
            if 'text' not in list(message.keys()):
                message['text'] = '<<텍스트아님>>'
                simplified_messages = [{'from_id': message['from']['id'],
                                        'text': message['text']}]
    print(simplified_messages)
    return last_update_id, simplified_messages


# 챗봇 API로 update_id 이후에 수신한 메시지를 조회하여 반환한다.
def get_updates(update_id):
    query = f'offset={update_id}'
    response = request_to_chatbot_api(method='getUpdates', query=query)
    return simplify_messages(response)


# 챗봇 API로 메시지를 chat_id 사용자에게 text 메시지를 발신한다.
def send_message(chat_id, text):
    text = urllib.parse.quote(text)
    query = f'chat_id={chat_id}&text={text}'
    response = request_to_chatbot_api(method='sendMessage', query=query)
    return response


# 챗봇으로 메시지를 확인하고, 적절히 응답한다.
def check_messages_and_response(next_update_id):
    last_update_id, recieved_messages = get_updates(next_update_id)  # ❶
    for message in recieved_messages:  # ❷
        chat_id = message['from_id']
        text = message['text']

        q = text
        a = ''
        while 1:
            if q == '<<텍스트아님>>':
                imoji = ["😂", "😍", "😚", "🤩", "🤔", "😉", "😻"] ## window키 .키
                a = random.choice(imoji)
                break
            elif q == "잘못했습니다":
                _, _, _, max_badbar = Warning_system("잘못했습니다", chat_id)
                break
            elif q == "/게이지":
                baduser, bad_bar, max_badbar, warning_message = Warning_system("/게이지", chat_id)
                if not bad_bar:
                    a = f"비속어 게이지 : □□□□□ {id_dict[chat_id]}/{max_badbar}"
                    break
                else:
                    a = f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
                    break
            elif q == "개발자 특권":
                baduser, bad_bar, max_badbar, a = Warning_system("개발자 특권", chat_id)
                break

            else:

                check = transform(q)
                input_ids = torch.tensor([check[0]]).to(device)
                token_type_ids = torch.tensor([check[1]]).to(device)
                attention_mask = torch.tensor([check[2]]).to(device)
                result = badword_model(input_ids, token_type_ids, attention_mask)
                idx = result.argmax().cpu().item()
                # print(idx)

                if idx == 1:
                    a = "성적표현 입니다. 경고가 누적됩니다."
                    break
                elif idx == 2:
                    a = "혐오표현 입니다. 경고가 누적됩니다."
                    break

                input_ids = torch.LongTensor(
                    koGPT2_TOKENIZER.encode("<usr>" + q + '<unused1>' + "<sys>" + a)).unsqueeze(dim=0)
                input_ids = input_ids.to(ctx)
                pred = chat_model(input_ids)
                pred = pred.logits
                pred = pred.cpu()
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == '</s>':
                    break
                a += gen.replace("▁", " ")

        if a == "개발자님 화이팅 충성충성 ^^7":
            send_text = a
        else:
            send_text = "{}".format(a.strip())  # ❸

        baduser, bad_bar, max_badbar, warning_message = Warning_system(send_text, chat_id)

        if baduser:
            if q == "잘못했습니다":
                _, _, _, warning_message = Warning_system("잘못했습니다", chat_id)
                send_message(chat_id, warning_message)
                print(f"챗봇>> {warning_message}")
                break
            else:
                send_message(chat_id, warning_message)
                print(f"챗봇>> {warning_message}")
                send_message(chat_id, "비매너 사용자에게는 응답하지 않습니다")
                print("챗봇>> 비매너 사용자에게는 응답하지 않습니다", )
        else:
            send_message(chat_id, send_text)  # ❹
            print(f"챗봇>> {send_text}")
            if send_text == '혐오표현 입니다. 경고가 누적됩니다.' or send_text == '성적표현 입니다. 경고가 누적됩니다.':
                send_message(chat_id, warning_message)
                print(f"챗봇>> {warning_message}")

        # send_message(chat_id, '당신의 매너수치는 ?')
    return last_update_id  # ❺


### 비속어 사용 시 경고 시스템 설정
def Warning_system(send_text, chat_id):
    global bad_bar
    max_sorry = 3  # 봐주는 횟수
    max_badbar = 3  # 최대욕설 횟수
    if chat_id not in id_dict:
        id_dict[chat_id] = 0  # 욕설 카운트
    if chat_id not in sorry_dict:
        sorry_dict[chat_id] = max_sorry  # 봐주는 횟수

    if send_text == '혐오표현 입니다. 경고가 누적됩니다.' or send_text == '성적표현 입니다. 경고가 누적됩니다.':
        if id_dict[chat_id] >= max_badbar:
            id_dict[chat_id] = max_badbar
        else:
            id_dict[chat_id] += 1
        print(id_dict[chat_id])
    elif send_text == "잘못했습니다":
        if (id_dict[chat_id] == 0) or (sorry_dict[chat_id] <= 0):
            print(id_dict[chat_id], sorry_dict[chat_id])
            return False, bad_bar, max_badbar, "더이상 봐드릴 수 없습니다"
        elif id_dict[chat_id] > 0:
            id_dict[chat_id] -= 1
            sorry_dict[chat_id] -= 1
            print(id_dict[chat_id], sorry_dict[chat_id])
            return False, bad_bar, max_badbar, "게이지 한칸만큼만 봐드리겠습니다."
    elif send_text == "개발자 특권":
        id_dict[chat_id] = 0
        sorry_dict[chat_id] = max_sorry
        bad_bar = f"{int(id_dict[chat_id]) * '■'}{(max_badbar - int(id_dict[chat_id])) * '□'}"
        print(id_dict[chat_id], sorry_dict[chat_id])
        return False, bad_bar, max_badbar, "개발자님 화이팅 충성충성 ^^7"

    bad_bar = f"{int(id_dict[chat_id]) * '■'}{(max_badbar - int(id_dict[chat_id])) * '□'}"

    if id_dict[chat_id] > max_badbar:
        id_dict[chat_id] = max_badbar
        if id_dict[chat_id] < max_badbar:
            baduser = False
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
        elif id_dict[chat_id] == max_badbar:
            baduser = True
            print(baduser)
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {max_badbar}/{max_badbar}"
    else:
        if id_dict[chat_id] < max_badbar:
            baduser = False
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
        elif id_dict[chat_id] == max_badbar:
            baduser = True
            print(baduser)
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {max_badbar}/{max_badbar}"

if __name__ == '__main__':  # ❶
    bot_token = '발급받은 텔레그램 챗봇 토큰'

    bot = telegram.Bot(token=bot_token)
    '''초기화'''
    # https://api.telegram.org/bot[자신의토큰]/getUpdates
    # bot.getUpdates("735781106")
    ''''''
    next_update_id = 0  # ❷
    id_dict = dict()
    sorry_dict = dict()

    while True:  # ❸
        last_update_id = check_messages_and_response(next_update_id)  # ❹
        if last_update_id:  # ❺
            next_update_id = last_update_id + 1
        time.sleep(1)  # ❻
        # npc = input("입력 : ")
        # if npc != "":
        #     chat_id = 5104167196
        #     bot.sendMessage(chat_id=chat_id, text=npc)

    '''원하는말 보내기'''
    # chat_id = 5104167196
    # bot.sendMessage(chat_id=chat_id, text="너무 이뻐요")