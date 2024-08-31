# decisionmodellib.py

## 디씨젼트리 모델 라이브러리

# 라이브러리 로딩
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# 데이터셋 로딩 함수
def importdata(filename):
    balance_data = pd.read_csv(
        filename,
        sep=',',
        header=None
    )
    print("데이터셋 길이 : ", len(balance_data))
    print("데이터셋 모양 : ", balance_data.shape)
    print("데이터셋 (상위 5행) : ", balance_data.head())
    return balance_data


# print(importdata(filename))

# 예측변수, 타겟변수 분리 함수
def splitdataset(balance_data, X, Y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X, Y, X_train, X_test, y_train, y_test


# 디씨젼트리분류기 생성 및 학습 함수
def train_using(criterion, random_state, md, msl, X_train, X_test, y_train):
    clf = DecisionTreeClassifier(
        criterion=criterion,  # 분류기의 종류(분류알고리즘)
        random_state=random_state,  # 랜덤시드값
        max_depth=md,  # 트리 뎁스
        min_samples_leaf=msl  # leaf노드 최소 개수
    )
    clf.fit(X_train, y_train)
    return clf


# 예측 함수
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("예측값 : ", y_pred)
    return y_pred


# 컨퓨젼매트릭스, 정확도, 레포트 출력 함수
def cal_accuracy(y_test, y_pred):
    print("컨퓨젼매트릭스 : ", confusion_matrix(y_test, y_pred))
    print("정확도 : ", accuracy_score(y_test, y_pred) * 100)
    print("레포트 : ", classification_report(y_test, y_pred))


# 디씨젼트리 가시화 함수
def plot_decision_tree(figsize, clf_object, feature_names, class_names):
    plt.figure(figsize=figsize)
    plot_tree(
        clf_object,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True
    )
    plt.show()


# 실행
filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
df = importdata(filename)
X = df.values[:, 1:5]  # 예측변수
Y = df.values[:, 0]  # 타겟변수
X, Y, X_train, X_test, y_train, y_test \
    = splitdataset(df, X, Y, 0.3, 1234)
clf = train_using('gini', 1234, 3, 5, X_train, X_test, y_train)
plot_decision_tree(
    (15, 10),
    clf,
    ['X1', 'X2', 'X3', 'X4'],
    ['L', 'B', 'R']
)