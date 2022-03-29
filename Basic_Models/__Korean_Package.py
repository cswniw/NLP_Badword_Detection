import re, jamotools, os, math
import sentencepiece as spm
from konlpy.tag import Okt, Mecab
from pykospacing import Spacing
### pip install git+https://github.com/haven-jeon/PyKoSpacing.git
from kobert_tokenizer import KoBERTTokenizer

from hanspell import spell_checker
### pip install git+https://github.com/ssut/py-hanspell.git


from konlpy.tag import Mecab
### 설치방법 https://lsjsj92.tistory.com/612

from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer

### 각 한국어 전처리 패키지를 함수화하였다.
### 첫번째 인자는 문자 데이터,
### 두번째 인자는 문자 데이터 구조가 데이터프레임이면 False, 스트링이면 True로 준다.


### 한스펠 한글 문법 교정 패키지
def DEF_Hanspell(X, predict=False) :
    if predict != True :
        for i in range(len(X)) :
            X[i] = spell_checker.check(X[i])
            X[i] = X[i].checked
        processed_X = X
        return processed_X
    else :
        X = spell_checker.check(X)
        processed_X = X.checked
        return processed_X


### 띄어쓰기 교정 패키지
def DEF_Spacing_Pyko(X, predict=False) :
    spacing = Spacing()
    if predict != True :
        for i in range(len(X)) :
            X[i] = spacing(X[i])
        processed_X = X
        return processed_X
    else :
        processed_X = spacing(X)
        return processed_X


### 은전한닢 패키지
def DEF_Mecab(X, predict=False) :  ### 설치방법 https://lsjsj92.tistory.com/612
    # m = MeCab.Tagger()
    # print(m.parse('안녕하세요. 이수진입니다.'))
    mecab = Mecab("C://mecab/mecab-ko-dic")
    if predict != True :
        for i in range(len(X)):
            X[i] = mecab.morphs(X[i])
        processed_X = X
        return processed_X
    else :
        # return mecab.morphs(X)
        return mecab.morphs(X)

### soynlp 패키지
def DEF_Soynlp(X, predict=False) :

    ### 패키지 초회차 사용시 단어집합을 구성하여 tokenizer를 생성해야한다.
    if 'soynlp_tokenizer.model' not in os.listdir('./pickled_ones') :
        file_path = 'datasets/data0303/spm_soynlp_texts_dataset.txt'
        corpus = DoublespaceLineCorpus(file_path)

        word_extractor = WordExtractor(min_frequency=10, min_cohesion_forward=0.05,
                                      min_right_branching_entropy=0.0)
        word_extractor.train(corpus)
        word_extractor.save('./pickled_ones/soynlp_tokenizer.model')

        # all cohesion probabilities was computed.  # words = 162890
        # all branching entropies was computed  # words = 506674
        # all accessor variety was computed  # words = 506674


    model_fname = "./pickled_ones/soynlp_tokenizer.model"
    word_extractor = WordExtractor(min_frequency=10, min_cohesion_forward=0.05,
                                   min_right_branching_entropy=0.0)
    word_extractor.load(model_fname)
    scores = word_extractor.word_scores()
    scores = {key: (scores[key].cohesion_forward * math.exp(scores[key].right_branching_entropy)) for key in
              scores.keys()}

    # 필요에 따라 LTokenizer 또는 MaxScoreTokenizer를 사용한다.
    # 비속어 데이터의 특성상 MaxScoreTokenizer가 더 적합하고, 실제로도 모델성능이 더 우수하게 나옴.

    tokenizer = MaxScoreTokenizer(scores=scores)        # tokenizer = LTokenizer(scores=scores)


    if predict != True :
        for i in range(len(X)) :
            X[i] = tokenizer.tokenize(X[i])
        processed_X = X
        return processed_X
    else :
        processed_X = tokenizer.tokenize(X)
        return processed_X


### OKT 품사 패키지
def DEF_Okt_Pos(X,predict=False):
    okt = Okt()
    if predict != True :
        for i in range(len(X)):
            # X[i] = re.sub('[^가-힣 ]', ' ', X[i])
            X[i] = okt.pos(X[i])
            words = []
            for j in X[i]:
                if (j[1] == 'Noun') or (j[1] == 'Verb') or (j[1] == 'Adjective') :
                # if (j[1] == 'Noun') or (j[1] == 'Verb') or (j[1] == 'Adjective') or (j[1] == 'Adverb') or (j[1] == 'Josa'):
                    words.append(j[0])
            X[i] = words
        processed_X = X
        return processed_X
    else:
        X = okt.pos(X)
        words = []
        for j in X:
            if (j[1] == 'Noun') or (j[1] == 'Verb') or (j[1] == 'Adjective'):
                words.append(j[0])
        processed_X = words
        return processed_X


### OKT 형태소 패키지
def DEF_Okt_Morphs(X,predict=False):
    okt = Okt()
    if predict != True :
        for i in range(len(X)):
            X[i] = okt.morphs(X[i], stem=True)
        processed_X = X
        return processed_X
    else:
        return okt.morphs(X, stem=True)


### 자모 분리 패키지
def DEF_Jamotools(X,predict=False) :
    # korean = re.compile('[^0-9!?^@ㄱ-ㅣ가-힣 ]+')
    # korean = re.compile('[^1!ㄱ-ㅣ가-힣 ]+')
    if predict != True:
        for i in range(len(X)):
            # X[i] = korean.sub('',jamotools.split_syllables(X[i]))
            X[i] = jamotools.split_syllables(X[i])
            # print(X[i])
            X[i] = list(X[i])
        processed_X = X
        return processed_X
    else:
        # processed_X = korean.sub('', jamotools.split_syllables(X))
        processed_X = jamotools.split_syllables(X)
        processed_X = list(processed_X)
        return processed_X

### 음절 단위 분리 함수
def DEF_Char(X, predict=False) :
    # korean = re.compile('[^0-9!?^@ㄱ-ㅣ가-힣 ]+')
    # korean = re.compile('[^1!ㄱ-ㅣ가-힣 ]+')
    if predict != True:
        for i in range(len(X)):
            # X[i] = korean.sub('', X[i])
            X[i] = list(X[i])
        processed_X = X
        return processed_X
    else:
        # processed_X = korean.sub('', X)
        # processed_X = list(processed_X)
        processed_X = list(X)
        return processed_X


### 구글 센텐스피스 초회 사용 시 서브워드 단어집합 구성을 위해 학습이 필요하다.
def DEF_Spm_set_training(X) :
    # with open('./data/spm_dataset.txt', 'w', encoding='utf8') as f:
    #     f.write('\n'.join(X))
    file = 'spm_soynlp_datasets_Mar07.txt'
    # file = 'spm_soynlp_texts_dataset.txt'
    spm.SentencePieceTrainer.Train('--input=./datasets/data0303/{} --model_prefix=./pickled_ones/spm_tokenizer_less '
                                   '--vocab_size=30000 --model_type=bpe --max_sentence_length=9999'.format(file))
    # 단어집합 사이즈는 통상 30000으로 한다. BPE 방식으로 단어집합을 구성한다.


### 구글 센텐스피스 패키지
def DEF_Sentencepiece(X, predict=False):
    if 'spm_tokenizer.model' not in os.listdir('./pickled_ones') :
        DEF_Spm_set_training(X)

    vocab_file = "./pickled_ones/spm_tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)

    if predict != True :
        for i in range(len(X)) :
            X[i] = sp.encode_as_ids(X[i])
        processed_X = X
        return processed_X
    else :
        processed_X = [sp.encode_as_ids(X)]
        return processed_X

    # encoded_ids = sp.encode_as_ids(i)
    # print(encoded_ids)
    # encoded_pieces = sp.encode_as_pieces(i)
    # print(encoded_pieces)
    # decoded_ids = sp.DecodeIds(encoded_ids)
    # print(decoded_ids)
    # decoded_pieces = sp.DecodePieces(encoded_pieces)
    # print(decoded_pieces)







# from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
#
# def get_tokenizer(tokenizer_name):
#     if tokenizer_name == "komoran":
#         tokenizer = Komoran()
#     elif tokenizer_name == "okt":
#         tokenizer = Okt()
#     elif tokenizer_name == "mecab":
#         tokenizer = Mecab()
#     elif tokenizer_name == "kkma":
#         tokenizer = Kkma()
#     else:
#         tokenizer = Mecab()
#     return tokenizer
#
# tokenizer = get_tokenizer("komoran")
# tokenizer.morphs("아버지가방에들어가신다")
#
# tokenizer = get_tokenizer("okt")
# tokenizer.morphs("아버지가방에들어가신다")
#
# tokenizer = get_tokenizer("mecab")
# tokenizer.morphs("아버지가방에들어가신다")
#
# tokenizer = get_tokenizer("kkma")
# tokenizer.morphs("아버지가방에들어가신다")

if __name__ == '__main__':
    DEF_Spm_set_training('')


    # print(DEF_Soynlp('', True))
    # print(DEF_Okt_Morphs('하이튼 인간말종 쓰레기 좀비 들이다 내한테 다온나 내가 싹쓸어줄께', True))
    # print(DEF_Okt_Pos('하이튼 인간말종 쓰레기 좀비 들이다 내한테 다온나 내가 싹쓸어줄께', True))

    # print(DEF_Hanspell('애비없산? x발개x같은새끼 사람죽겄다 역겹네너를 왜까겠냐?', True))
    # print(DEF_Sentencepiece('하이튼 인간말종 쓰레기 좀비 들이다 내한테 다온나 내가 싹쓸어줄께',True))

    # sentence = 'ㅅㅂ...ㅄ...ㅇㅂ새끼들   ㅈㄲ라 그래 병신들 외그따구로 삼?'  ## 1번
    # sentence = '대깨문들 꼴페미랑 짝짜꿍쳐서 개꼴 ㅋ ㅈㄴ 싸고싶네 ㅇㅂㅅㄲ 박멸 가즈아~' ## 2번
    # sentence = '느그음마~~느금마 느그으애미 씹꼴리네 ㅇㅂ ㅄ ㄲㅈ ㄱㅐ새ㄲㅣ덜~ 가서쳐드셈 ' ## 3번
    # sentence = '없어진듯문재앙 씨발개좆같은새끼 그만좀해라사람죽겄다 역겹네보겸을 왜까겠냐' ## 4번
    # sentence = 'ㅗ 이거나 드셈 ㅗㅡㅡㅗ'




    # https://keep-steady.tistory.com/7
    # print(DEF_Sentencepiece(sentence, True))
    ### 정수 엔코딩된 숫자로 나오니 테스트 시 아래의 코드 사용.
    ### 1번. ['▁ᄉᄇ', '...', 'ᄡ', '...', 'ᄋᄇ', '새끼들', '▁ᄌ', 'ᄁ', '라', '▁그래', '▁병신들', '▁외', '그', '따', '구로', '▁삼', '?']
    ### 2번. ['▁대깨문들', '▁꼴페미', '랑', '▁짝', '짜', '꿍', '쳐서', '▁개', '꼴', '▁ᄏ', '▁ᄌᄂ', '▁싸', '고싶', '네', '▁ᄋᄇ', 'ᄉᄁ', '▁박', '멸', '▁가즈아', '~']
    ### 3번. ['▁느그', '음', '마', '~~', '느', '금', '마', '▁느그', '으', '애미', '▁씹', '꼴', '리네', '▁ᄋᄇ', '▁', 'ᄡ', '▁ᄁ', 'ᄌ', '▁개새끼', '덜', '~', '▁가서', '쳐', '드', '셈']
    ### 4번. ['▁없어', '진', '듯', '문재앙', '▁씨발', '개', '좆', '같은', '새끼', '▁그만좀', '해라', '사람', '죽', '겄다', '▁역겹네', '보', '겸', '을', '▁왜', '까', '겠냐']
    ### 5번. ['▁ᅩ', '▁이거', '나', '▁드', '셈', '▁ᅩ', 'ᅳᅳ', 'ᅩ']
    ### KoGPT2
    ### 1번. ['▁ᄉ', 'ᄇ', '...', 'ᄡ', '...', 'ᄋ', 'ᄇ', '새', '끼', '들', '▁', '▁', '▁', 'ᄌ', 'ᄁ', '라', '▁그래', '▁병', '신', '들', '▁외', '그', '따', '구로', '▁삼', '?']
    ### 2번. ['▁대', '깨', '문', '들', '▁꼴', '페', '미', '랑', '▁짝', '짜', '꿍', '쳐서', '▁개', '꼴', '▁', 'ᄏ', '▁', 'ᄌ', 'ᄂ', '▁싸고', '싶', '네', '▁', 'ᄋ', 'ᄇ', 'ᄉ', 'ᄁ', '▁박', '멸', '▁가', '즈', '아', '~']
    ### 3번. ['▁느', '그', '음', '마', '~', '~', '느', '금', '마', '▁느', '그', '으', '애', '미', '▁씹', '꼴', '리', '네', '▁', 'ᄋ', 'ᄇ', '▁', 'ᄡ', '▁', 'ᄁ', 'ᄌ', '▁개', '새', '끼', '덜', '~', '▁가서', '쳐', '드', '셈', '▁']
    ### 4번. ['▁없어진', '듯', '문', '재', '앙', '▁씨', '발', '개', '좆', '같은', '새', '끼', '▁그만', '좀', '해', '라', '사람', '죽', '겄', '다', '▁역', '겹', '네', '보', '겸', '을', '▁왜', '까', '겠', '냐']

    # print(DEF_Mecab(sentence, True))

    ### 1번. ['ㅅ', 'ㅂ', '.', '..', 'ㅄ', '.', '..', 'ㅇㅂ새끼들', 'ㅈ', 'ㄲ라', '그래', '병신', '들', '외', '그', '따구', '로', '삼', '?']
    ### 2번. ['대', '깨', '문', '들', '꼴', '페미', '랑', '짝짜꿍', '쳐서', '개꼴', 'ㅋ', 'ㅈ', 'ㄴ', '싸', '고', '싶', '네', 'ㅇㅂㅅㄲ', '박멸', '가즈', '아', '~']
    ### 3번. ['느그', '음', '마', '~~', '느', '금마', '느그', '으', '애미', '씹', '꼴리', '네', 'ㅇㅂ', 'ㅄ', 'ㄲㅈ', 'ㄱ', 'ㅐ새ㄲㅣ덜', '~', '가', '서', '쳐', '드셈']
    ### 4번. ['없', '어', '진', '듯', '문재', '앙', '씨발', '개', '좆같', '은', '새끼', '그만', '좀', '해라', '사람', '죽', '겄', '다', '역겹', '네', '보', '겸', '을', '왜', '까', '겠', '냐']
    ### 5번. ['ㅗ', '이거', '나', '드셈', 'ㅗㅡㅡㅗ']
    # print(DEF_Soynlp(sentence, True))
    ### 1번. ['ㅅㅂ', '...ㅄ...ㅇㅂ새끼들', 'ㅈㄲ라', '그래', '병신', '들', '외그따구로', '삼?']
    ### 2번. ['대깨문들', '꼴페미', '랑', '짝짜꿍쳐서', '개꼴', 'ㅋ', 'ㅈㄴ', '싸고싶네', 'ㅇㅂ', 'ㅅㄲ', '박멸', '가즈아', '~']
    ### 3번. ['느그', '음마~~느금마', '느그', '으애미', '씹꼴리네', 'ㅇㅂ', 'ㅄ', 'ㄲㅈ', 'ㄱㅐ새ㄲㅣ덜~', '가서', '쳐드셈']
    ### 4번. ['없어', '진듯문재앙', '씨발', '개좆같은새끼', '그만', '좀해라사람죽겄다', '역겹', '네보겸을', '왜까겠냐']
    ### 5번. ['ㅗ', '이거', '나', '드셈', 'ㅗㅡㅡㅗ']
    # print(DEF_Okt_Morphs(sentence, True))
    ### 1번. ['ㅅㅂ', '...', 'ㅄ', '...', 'ㅇㅂ', '새끼', '들', 'ㅈㄲ', '라', '그렇다', '병신', '들', '외그', '따다', '삼', '?']
    ### 2번. ['대', '깨문', '들', '꼴', '페미', '랑', '짝', '짜다', '꿍', '치다', '개꼴', 'ㅋ', 'ㅈㄴ', '싸다', 'ㅇㅂㅅㄲ', '박멸', '가즈', '아', '~']
    ### 3번. ['느그', '음마', '~~', '느금마', '느그', '으애', '밉다', '씹다', '꼴리다', 'ㅇㅂ', 'ㅄ', 'ㄲㅈ', 'ㄱㅐ', '새', 'ㄲㅣ', '덜', '~', '가다', '치다', '드세다']
    ### 4번. ['없어지다', '문', '재앙', '씨발', '개', '좆', '같다', '새끼', '그만', '좀해', '라', '사람', '죽겄', '다', '역겹다', '네', '보겸', '을', '왜', '끄다']
    ### 5번. ['ㅗ', '이', '거나', '드세다', 'ㅗㅡㅡㅗ']
    # print(DEF_Okt_Pos(sentence, True))
    ### 1번. ['새끼', '그래', '병신', '외그', '따구로', '삼']  # 자음이 아예 삭제.
    ### 2번. ['깨문', '꼴', '페미', '짝', '짜', '꿍', '쳐서', '개꼴', '싸고싶네', '박멸', '가즈']
    ### 3번. ['음마', '느금마', '으애', '미', '씹', '꼴리네', '새', '덜', '가서', '쳐', '드셈']
    ### 4번. ['없어진듯', '재앙', '씨발', '좆', '같은', '새끼', '좀해', '사람', '죽겄', '역겹', '보겸', '왜', '까겠냐']
    ### 5번. ['거나', '드셈']
    # print(DEF_Hanspell(sentence,True))
    ### 1번. ㅅㅂ...ㅄ...ㅇㅂ새끼들   ㅈㄲ라 그래 병신들 외그따구로 삼?
    ### 2번. 대 깨문들 꼴페미랑 짝짜꿍 쳐서 개꼴 ㅋ 전 싸고 싶네 ㅇㅂㅅㄲ 박멸 가즈에~
    ### 3번. 느그음마~~느금마 느그으애미 씹꼴리네 업 ㅄ ㄲㅈ ㄱㅐ새ㄲㅣ덜~ 가서 쳐드세요
    ### 4번. 없어진 듯 문재원 씨발개좆같은새끼 그만 좀 해라 사람 죽겠다 역겹네 보 겸을 왜 까겠냐
    ### 5번. ㅗ 이거나 드세요 ㅗㅡㅡㅗ
    # print(DEF_Spacing_Pyko(sentence,True))
    ### 1번. ㅅㅂ...ㅄ...ㅇㅂ 새끼들 ㅈㄲ라 그래 병신들 외 그따구로 삼?
    ### 2번. 대깨문들 꼴 페미랑 짝짜 꿍쳐서 개꼴 ㅋ ㅈㄴ 싸고 싶네 ㅇㅂㅅㄲ 박멸 가즈아~
    ### 3번. 느 그음마~~느금마 느 그으애미 씹꼴리네 ㅇㅂ ㅄ ㄲㅈ ㄱㅐ 새 ㄲㅣ 덜~ 가서 쳐드셈
    ### 4번. 없어진 듯 문재앙 씨 발개좆 같은 새끼 그만 좀 해라 사람 죽 겄다 역겹네보 겸을 왜까겠냐
    ### 5번. ㅗ 이거나 드셈 ㅗㅡㅡㅗ
    ### Kobert 토크나이저 ###
    # tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    # add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    #              'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
    # tokenizer.add_tokens(add_token)
    # print(tokenizer.tokenize(sentence))
    ### 1번. ['ㅅ', 'ㅂ', '▁', '...', 'ㅄ', '▁', '...', 'ㅇ', 'ㅂ', '▁새', '끼', '들', 'ㅈ', 'ㄲ', '▁라', '▁', '그래', '▁병', '신', '들', '▁외', '그', '따', '구', '로', '▁삼', '?']
    ### 2번. ['▁대', '깨', '문', '들', '▁', '꼴', '페', '미', '랑', '▁', '짝', '짜', '꿍', '쳐', '서', '▁개', '꼴', 'ㅋ', 'ㅈ', 'ㄴ', '▁싸', '고', '싶', '네', 'ㅇ', 'ㅂ', 'ㅅ', 'ㄲ', '▁박', '멸', '▁', '가', '즈', '아', '~']
    ### 3번. ['▁', '느', '그', '음', '마', '~', '~', '느', '금', '마', '▁', '느', '그', '으', '애', '미', '▁', '씹', '꼴', '리', '네', 'ㅇ', 'ㅂ', 'ㅄ', 'ㄲ', 'ㅈ', 'ㄱ', '▁', 'ᅢ', '새', 'ㄲ', '▁', 'ᅵ', '덜', '~', '▁', '가', '서', '쳐', '드', '셈']
    ### 4번. ['▁없어', '진', '듯', '문', '재', '앙', '▁씨', '발', '개', '좆', '같은', '새', '끼', '▁그', '만', '좀', '해', '라', '사람', '죽', '겄', '다', '▁역', '겹', '네', '보', '겸', '을', '▁왜', '까', '겠', '냐']
    ### 5번. ['▁', 'ᅩ', '▁이', '거나', '▁드', '셈', '▁', 'ᅩᅳᅳᅩ']

    ### 센텐스 피스 쓰려면 이걸로 ####
    sentence = '하이튼 인간말종 쓰레기 좀비 들이다 내한테 다온나 내가 싹쓸어줄께'

    vocab_file = "./pickled_ones/spm_tokenizer.model"
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)
    encoded_pieces = sp.encode_as_pieces(sentence)
    print(encoded_pieces)




    # b = ['해물탕이뭐어때서 ㅋㅋ 아들이름이 베터리던데 문제인']
    # print(DEF_Spacing_Pyko(b))
    # b = ['해물탕이 뭐어때서 ㅋㅋ 아들 이름이 배터리던 데 문제인']
    # print(DEF_Mecab(b))




    # # encoded_ids = sp.encode_as_ids(i)
    # # print(encoded_ids)
    # encoded_pieces = sp.encode_as_pieces(i)
    # print(encoded_pieces)
    # decoded_ids = sp.DecodeIds(encoded_ids)
    # print(decoded_ids)
    # decoded_pieces = sp.DecodePieces(encoded_pieces)
    # print(decoded_pieces)

    pass
