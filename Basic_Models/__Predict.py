import glob
import sentencepiece as spm
from tensorflow.keras.models import Sequential, load_model
from __NLP_Dataloader import *
from __Korean_Package import *


### 모델 평가를 위해 첫번째 문자데이터, 두번째 불러올 모델의 이름, 세번쨰 전처리 패키지를 인자로 준다.
### 해당하는 tokenizer 또는 encoder 등 파일명으로부터 필요한 시퀀스 길이, 단어집합 크기의 정보를 추출하여
### 모델 평가 시 바로적용할 수 있도록 자동화하였다.

def DEF_Badlang_Predict(new_sentence, model_name, korean_package):


    loaded_model = load_model(f'./saved_model/{korean_package}_{model_name}_best_model.h5')

    with open(f'pickled_ones/{korean_package}_encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
    label = encoder.classes_

    if korean_package == 'spm':
        pass
    else :
        with open(f'pickled_ones/{korean_package}_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)

    target = f'./saved_np/{korean_package}_badlang_dataset_*_*.npy'
    find_vocab_size = glob.glob(target)[0][:-4].split('_')  # 단어집합 크기 찾기
    Max_len, Vocab_size = int(find_vocab_size[-2]), int(find_vocab_size[-1])

    if korean_package != 'spm' :
        if korean_package == 'okt_morphs' :
            # new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
            new_sentence = DEF_Okt_Morphs(new_sentence, True)  # 토큰화
            new_sentence = DEF_Stopwords(new_sentence, True)  # 불용어 제거
        elif korean_package == 'okt_pos' :
            # new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
            new_sentence = DEF_Okt_Pos(new_sentence, True)  # 토큰화
        elif korean_package == 'jamo' :
            new_sentence = DEF_Jamotools(new_sentence, True)
        elif korean_package == 'char':
            new_sentence = DEF_Char(new_sentence, True)
        elif korean_package == 'mecab':
            new_sentence = DEF_Mecab(new_sentence, True)
        elif korean_package == 'soynlp':
            new_sentence = DEF_Soynlp(new_sentence, True)

        encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩

    elif korean_package == 'spm':
        encoded = DEF_Sentencepiece(new_sentence, True)


    pad_new = pad_sequences(encoded, maxlen=Max_len)  # 패딩
    # print(pad_new)
    # (1,85)
    preds = loaded_model.predict(pad_new)  # 예측
    predicts = (label[np.argmax(preds)])

    return predicts, preds



if __name__ == '__main__':

    # 전처리 패키지 선택
    korean_package_list = ['okt_morphs', 'okt_pos', 'jamo', 'char', 'spm', 'mecab', 'soynlp']
    korean_package = korean_package_list[5]

    # 모델 선택
    model_name_list = ['LSTM','GRU','Bi_LSTM','onedCNN']
    model_name = model_name_list[3]


    # input에 문자 데이터 입력 시 예측 분류값과 확률을 출력한다.
    while True :
        question = input('입력하세요 >>> ')
        if question == '즐':
            break
        answer, pred = DEF_Badlang_Predict(question, model_name, korean_package)
        print(answer, pred)