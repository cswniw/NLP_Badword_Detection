import pandas as pd
from sklearn.model_selection import train_test_split


####################  데이터 스플릿   #######################
df = pd.read_csv('datasets/data0303/concat_datasets_0303.csv', sep="|")

train_df = pd.DataFrame()
test_df = pd.DataFrame()

categories = ['혐오', '섹슈얼', '일반']

for i in categories:
    filter_df = df[df['카테고리'] == i]
    train_set, test_set = train_test_split(filter_df, test_size=0.2)  # test_size = 비율설정
    train_df = pd.concat([train_df, train_set], ignore_index=True)
    test_df = pd.concat([test_df, test_set], ignore_index=True)

# 행 섞어주기
test_df = test_df.sample(frac=1).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_df.to_csv("./datasets0225/data0303/concat_evaluation_datasets_0303.csv",sep='|', index=False)
train_df.to_csv("./datasets0225/data0303/concat_train_test_datasets_0303.csv", sep='|',index=False)


####################  업 샘플링   #######################
# 각 카테고리 수가 불균형하므로 필요 시 업 샘플링을 진행한다.

file_path = "datasets/data0303/concat_train_test_datasets_0303.csv"
df = pd.read_csv(file_path, sep='|')

df['카테고리'].value_counts()
# 데이터가 가장 많은 카테고리를 찾는다.
longlong = len(df[df['카테고리'] == '일반'])
# print(longlong) ## 2248
test_df_1 = pd.DataFrame()

### 부족한 카테고리의 배수 만큼 데이터를 복사하고 부족한 숫자는 비복원 추출한다.
### 예시. 일반 카테고리 2000개/ 혐오 카테고리 900개
### 혐오 카테고리 2000 - (900x2) = 200          ~> 2번 데이터 복사 후 900 중 200개 비복원 추출

for i in ['일반', '섹슈얼','혐오']:

    if (longlong - len(df[df['카테고리'] == i])) >= 0 :

        df_1 = (df[df['카테고리'] == i])

        y = df_1.sample(n=(longlong%len(df_1)), replace=False)
        test_df_1 = pd.concat([test_df_1,y])

        test_df_2 = []
        for j in range(longlong//len(df_1)) :
            for k in range(len(df_1)) :
                test_df_2.append(df_1.iloc[k, :])
        df_2 = pd.DataFrame(test_df_2)
        test_df_1 = pd.concat([test_df_1, df_2])

test_df_1.info()
print(test_df_1['카테고리'].value_counts())
test_df_1.to_csv("./datasets0225/data0303/upsampling_concat_train_test_datasets_0303.csv",
                 sep='|', index=False)


