## boosting.py
# 부스팅 기법

def p(str):
    print(str, '\n')

# 라이브러리 로딩
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# iris 데이터 로딩
iris = load_iris()
X = iris.data
y = iris.target

# 트레인, 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Adaptive Boosting

# base_estimator : 기본 분류기 지정 (기본 값이 의사 결정 트리)
# max_depth : 트리 깊이
# n_estimators : 부스팅을 수행할 기본 분류 기의 개수 (기본 값이 50)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50
)
# p(ada)

# 모델 학습
ada.fit(X_train, y_train)

# 예측값
ada_pred = ada.predict(X_test)

# 정확도
ada_acc = accuracy_score(y_test, ada_pred)
p(ada_acc)


## gradient Boosting

# n_estimators : 부스팅을 수행할 기본 분류기 개수, 기본 값 100
# learning_rate : 학습률, 기본 값: 0.1, 학습률이 작을 수록 모델을 안정적으로 만들고
#                 과적합을 줄일 수 있다.
# max_depth : 트리의 최대 깊이, 기본 값은 3이다.
gb = GradientBoostingClassifier(
    n_estimators = 365,
    learning_rate = 0.1,
    max_depth = 1,
)
p(gb)

# 모델 학습
gb.fit(X_train, y_train)

# 예측값
gb_pred = gb.predict(X_test)

# 정확도
gb_acc = accuracy_score(y_test, gb_pred)
p(gb_acc)
# 이 모델에서는 ada가 정확한 것을 확인할 수 있다.

## 트리 시각화

# 라이브러리 로딩
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

# 데이터 로딩
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 데이터 프레임 생성
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# p(df)

df['label'] = [iris.target_names[x] for x in iris.target]

# 독립 변수/종속 변수 생성
X = df.drop('label', axis=1)
y = df['label']

# 모델 생성
clf = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=3,
    min_samples_leaf=5
)

# 모델 학습
clf.fit(X, y)

# 예측 값 출력
clf_pred = clf.predict(X)[:3]
p(clf_pred)

# 변수 중요도 측정
for i, column in enumerate(X.columns):
    p(f'{column} 중요도: {clf.feature_importances_[i]}')

# 정확도
p(f'정확도 : {clf.score(X, y)}')

# 시각화
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_)
plt.show()

# clf 정보 확인 함수
import numpy as np
def get_info(dt_model, tree_type='clf'):
    tree = dt_model.tree_
    # 분할에 사용된 기준
    criterion = dt_model.get_params()['criterion']
    # 트리 유형이 유효한지 확인
    # assert : 주어진 조건이 참이 아니면 AssertionError를 발생시킨(디버깅용으로 사용)
    assert tree_type in ['clf', 'reg']
    # 트리의 노드 수
    num_node = tree.node_count
    # 노드 정보를 저장할 리스트
    info = []
    # 트리의 각 노드 반복
    for i in range(num_node):
        # 각 노드의 정보를 저장할 딕셔너리
        temp_di = dict()
        # 현재 노드가 분할을 나타내는지 확인
        if tree.threshold[i] != -2: # -2:leaf node
            # 분할에 사용된 특징과 임계값 저장
            split_feature = tree.feature[i]
            split_thres = tree.threshold[i]
            # 분할 질문
            temp_di['question'] = f'{split_feature} <= {split_thres:.3f}'
            # 불순도와 노드에 포함된 샘플 수
            impurity = tree.impurity[i]
            sample = tree.n_node_samples[i]
            # 불순도와 샘플 수 저장
            temp_di['impurity'] = f'{criterion} = {impurity:.3f}'
            temp_di['sample'] = sample
            # 예측된 값(회귀), 클래스 확률(분류)
            value = tree.value[i]
            temp_di['value'] = value
            # 분류 트리의 경우 예측된 클래스 레이블 저장
            if tree_type == 'clf':
                classes = dt_model.classes_
                idx = np.argmax(value)
                temp_di['class'] = classes[idx]
        # 노드 정보를 리스트에 추가
        info.append(temp_di)
    return info

# 함수 실행
p(get_info(clf))


## Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=1
)

# 학습
gb.fit(X_train, y_train)

first = gb.estimators_[0][0]

plt.figure(figsize=(8, 5))
plot_tree(first, filled=True, feature_names=iris.feature_names)
plt.show()


## 회귀 나무 (DecisionTreeRegressor)
from sklearn import datasets
X, y = datasets.fetch_openml('boston', return_X_y = True)

# 보스턴 주택 가격 데이터
df = pd.DataFrame(X)
df['MEDV'] = y
p(df)

# 독립변수, 종속 변수 분리
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=3,
    min_samples_leaf=10,
    random_state=100
)

# 학습
reg.fit(X, y)

# 예측값
p(reg.predict(X)[:3])

# 변수 중요도
for i, column in enumerate(X.columns):
    p(f'{column} 중요도 : {reg.feature_importances_[i]}')

# 트리 시각화
plt.figure(figsize=(15, 12))
plot_tree(reg, feature_names=X.columns)
plt.show()

# 트리의 노드 정보
p(get_info(reg, tree_type='reg'))







































































































