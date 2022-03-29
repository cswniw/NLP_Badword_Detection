import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, classification_report
from __Predict import *

### 모델의 성능을 평가하는 함수
### 정답과 예측값을 비교하여 confusion matrix / 정확도,정밀도,민감도,F-1 Score 정보를 확인할 수 있도록 함수화함.

def DEF_Evaluate(korean_package, model_name) :

    # 평가용 데이터셋 불러오기
    df = pd.read_csv('datasets/data0303/concat_evaluation_datasets_0303.csv', sep="|")
    X, Y = df['text'].to_list(), df['카테고리'].to_list()

    # 해당 모델과 사용한 전처리 패키지 불러오기
    model = load_model(f'./saved_model/{korean_package}_{model_name}_best_model.h5')
    # 엔코더 불러오기
    with open(f'pickled_ones/{korean_package}_encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)
    label = encoder.classes_



    if f'{korean_package}_{model_name}_Z_test.pickle' not in os.listdir('./evaluation_report'):

        if korean_package == 'spm':
            pass
        else :
            with open(f'pickled_ones/{korean_package}_tokenizer.pickle', 'rb') as f:
                tokenizer = pickle.load(f)

        ### 해당 시퀀스 길이, 단어집합 크기 추출.
        target = f'./saved_np/{korean_package}_badlang_dataset_*_*.npy'
        find_vocab_size = glob.glob(target)[0][:-4].split('_')
        Max_len, Vocab_size = int(find_vocab_size[-2]), int(find_vocab_size[-1])

        ### 해당 전처리 패키지로 예측할 문자데이터 가공.
        if korean_package != 'spm':
            if korean_package == 'okt_morphs' :
                processed_X = DEF_Okt_Morphs(X)
                processed_X_with_stopped = DEF_Stopwords(processed_X)   ## 일단 이것만 스탑워드 함수 사용.
            elif korean_package == 'okt_pos' :
                processed_X = DEF_Okt_Pos(X)
                processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
            elif korean_package == 'jamo' :
                processed_X = DEF_Jamotools(X)
                processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
            elif korean_package == 'char' :
                processed_X = DEF_Char(X)
                processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
            elif korean_package == 'mecab' :
                processed_X = DEF_Mecab(X)
                processed_X_with_stopped = processed_X
            elif korean_package == 'soynlp' :
                processed_X = DEF_Soynlp(X)
                processed_X_with_stopped = processed_X

            tokened_X = DEF_Tokenizing(processed_X_with_stopped, Vocab_size, korean_package)


        elif korean_package == 'spm':
            processed_X = DEF_Sentencepiece(X)
            tokened_X = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드


        ## 패딩
        X_pad = DEF_Padding_Sequences(tokened_X, Max_len)
        Z = []
        # print(X_pad.shape)  # (1289,85)

        # a = X_pad[0]
        # b = np.reshape(a, (1,85))

        ### 예측값 저장.
        for i in range(len(X_pad)):
            a = X_pad[i]
            b = np.reshape(a, (1,Max_len))
            preds = model.predict(b)
            predicts = (label[np.argmax(preds)])
            Z.append(predicts)
            if i % 100 == 0:
                print("★", end=" ")

        print()
        with open(f'./evaluation_report/{korean_package}_{model_name}_Z_test.pickle', 'wb') as f:
            pickle.dump(Z, f)


    else :
        with open(f'./evaluation_report/{korean_package}_{model_name}_Z_test.pickle', 'rb') as f:
            Z = pickle.load(f)


    '''정답,예측 비교'''
    # df_a = pd.DataFrame({"text": X, "정답": Y, "예측": Z})
    # df_a.to_csv(f'./evaluation_report/정답비교_{korean_package}_{model_name}.csv', index=False)
    # print(f"============================================{df_a}==================================")

    '''예측 잘못된 데이터들 확인'''
    # target_names = ['섹슈얼', '일반_긍정', '혐오']

    # for a1 in label:
    #     for a2 in label:
    #         if a1 != a2:
    #             error_df = []
    #             for i in range(len(X)):
    #                 if Y[i] == a1 and Z[i] == a2:
    #                     error_df.append(X[i])
    #             error_df = pd.Series(error_df, name=f"{a1}->{a2}")
    #             print(error_df)
    #             error_df.to_csv(f'./evaluation_report/{korean_package}_{model_name}_{a1}_{a2}_error_data.csv',
    #                             index=False)

    '''성능평가 데이터'''
    print(confusion_matrix(Y, Z))
    print(classification_report(Y, Z, target_names=label))
    print(korean_package, model_name)

if __name__ == '__main__':

    korean_package_list = ['okt_morphs', 'okt_pos', 'jamo', 'char', 'spm', 'mecab', 'soynlp']
    korean_package = korean_package_list[2]

    model_name_list = ['LSTM', 'GRU', 'Bi_LSTM', 'onedCNN']
    model_name = model_name_list[3]

    DEF_Evaluate(korean_package,model_name)


    # for i in model_name_list :
    #     for j in korean_package_list :
    #         if j == 'soynlp' :
    #             DEF_Evaluate(j, i)


    # for model_name in model_name_list:

    # '''첫 실행시만 주석 해제'''
    # # for i in range(len(X)):
    # #     Z.append(DEF_Badlang_Predict(X[i], f'{model_name}', f'{korean_package}'))
    # #     if i % 100 == 0:
    # #         print("★", end=" ")
    # # print()
    #
    #
    # with open(f'./evaluation_report/{korean_package}_{model_name}_Z_test.pickle', 'wb') as f:
    #     pickle.dump(Z, f)
    #
    # '''이전 파일 읽어올때 주석 해제'''
    # # with open(f'./pickled_ones/{korean_package}_{model_name}_Z_test.pickle', 'rb') as f:
    # #     Z = pickle.load(f)
    #

