## decisiontree.py

def p(str):
    print(str, '\n')

## 소득 예측 모델 만들기
import pandas as pd
df = pd.read_csv('./assets/adult.csv')
df.info()
# 위의 csv 파일에서 income이 종속변수이고, 나머지가 독립변수이다.
p('')

# 데이터 전처리
import numpy as np

# 연소득 5만달러 초과하면 high, 그렇지 않으면 low로
df['income'] = np.where(df['income'] == '>50K', 'high', 'low')
p(df['income'].value_counts(normalize=True)) # normalize=True 범주 비율로 출력하는 방법

# 타켓변수(종속변수) 예측에 도움이 안되는 불필요한 변수 제거
df = df.drop(columns='fnlwgt')

# 원핫인코딩(one-hot encoding)을 이용한 문자타입 변수를 숫자 타입으로 변환
# 원핫 인코딩: 데이터를 모두 1이나 0으로 표시

target = df['income']
df = df.drop(columns='income') # 종속변수 제거
df = pd.get_dummies(df)
df['income'] = target
df.info(max_cols=np.inf) # 변수의 수에 관계없이..

# scikit-learn을 이용하여 트레이닝 세트와 테스트 세트 분리
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df,
                                     test_size=0.3,             # 테스트 세트 비율
                                     stratify=df["income"],     # 범주별 비율을 통일할 변수
                                     random_state=1234)         # 난수 초기값
p(df_train.shape)   # 트레이닝 세트의 행 개수, 변수 개수
p(df_test.shape)    # 테스트 세트의 행 개수, 변수 개수

# 트레이닝 세트와 테스트 세트의 범주별 비율
p(df_train["income"].value_counts(normalize=True))
p(df_test["income"].value_counts(normalize=True))

## 의사 결정 나무 모델 생성

# 라이브러리 임포트
from sklearn import tree

# 의사결정나무 뷴류기 생성
# random_state: 랜덤 시드값, max_depth: 트리의 최대 깊이
clf = tree.DecisionTreeClassifier(random_state=1234, max_depth=3)

# 예측변수, 타겟변수 추출
train_x = df_train.drop(columns='income') # 예측 변수 (전체 15개 변수 중 타겟 변수를 제외한 것)
train_y = df_train['income'] # 타겟 변수 (예측 변수를 통해 예측하고자 하는 변수)

# 의사결정 나무 모델 만들기
# fit(X : 예측 변수, y : 타겟 변수)
model = clf.fit(X=train_x, y=train_y) # 각각 예측 변수와 타겟 변수

# 모델 시각화
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        'figure.dpi' : 100, # 그래프 해상도
        'figure.figsize' : [12, 8] # 그래프 크기
    }
)

# 트리 그래프
tree.plot_tree(
    model, #모델
    feature_names=train_x.columns, # 예측 변수의 명칭들
    class_names = ['high', 'low'], # 타겟 변수 클래스(알파벳 오름순)
    proportion = True, # 비율 표시 여부
    filled = True, # 채우기 여부
    rounded = True, # 노드 테두리를 둥글게 할지 여부
    impurity = False, # 불순도 표시 여부
    label = 'root', # 제목 표시 위치
    fontsize = 10 # 글자 크기
)
plt.show()
# 옵션들을 바뀌가며 그려보기

# 예측을 위한 예측 변수, 타겟 변수 추출
test_x = df_test.drop(columns='income') # 예측 변수
test_y = df_test['income'] # 타겟 변수

# 모델을 통한 예측
df_test['pred'] = model.predict(test_x)
p(df_test)

## 예측 성능 평가

# 컨퓨전 메트릭스 생성
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(
    y_true = df_test['income'], # 실제 값
    y_pred = df_test['pred'], #예측 값
    labels = ['high', 'low'] #레이블 (클래스 배치 순서, 문자 오름차순)
)
p(conf_matrix)

## 컨퓨전 메트릭스를 히트맵으로 표시

# 그래스 설정 되돌리기
plt.rcdefaults() # 그래프 설정 초기화
from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, # 메트릭스 데이터
    display_labels = ('high', 'low') # 타겟변수 클래스명
)

p.plot(cmap = 'Blues') # 컬러맵
plt.show()

## 성능평가 지표 구하기

import sklearn.metrics as metrics

# Accuracy : 정확도
acc = metrics.accuracy_score(
    y_true = df_test['income'], # 실제 값
    y_pred = df_test['pred'] # 예측 값
)
print(acc)
# 0.8439227461953184

# Precision : 정밀도
pre = metrics.precision_score(
    y_true = df_test['income'], # 실제 값
    y_pred = df_test['pred'], # 예측 값
    pos_label = 'high' # 관심 클래스
)
print(pre)
# 0.7557700377675199

# Recall : 재현율
rec = metrics.recall_score(
    y_true = df_test['income'], # 실제 값
    y_pred = df_test['pred'], # 예측 값
    pos_label = 'high' # 관심 클래스
)
print(rec)
# 0.5136908157444381

# F1 Score
f1 = metrics.f1_score(
    y_true = df_test['income'], # 실제 값
    y_pred = df_test['pred'], # 예측 값
    pos_label = 'high' # 관심 클래스
)
print(f1)
# 0.6116488368143997
# f1 값은 재현율과 정밀도 중간정도로 나온다. 평균은 아니다.
























