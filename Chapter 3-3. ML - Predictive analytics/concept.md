# English

## Machine Learning Models

Machine learning models are used to predict future data based on the data available up to the present.

- **Predictor Variables**
    
    → Variables used to make predictions or the inputs to the model.
    
- **Target Variables**
    
    → Variables that the model aims to predict or the outputs of the model.
    
The model is used to find the **target variable** through **predictor variables**.

In the expression $y = f(x)$, $y$ is the target variable, and $f(x)$ is the predictor variable.

## Decision Tree Model

- A model that predicts by sequentially acquiring binary "Yes/No" answers.
- Its simple structure and ease of understanding make it the foundation for many prediction models.
- It focuses on the predictor variables that best separate the target variable.

    For example, among predictors like age, smoking, and alcohol consumption, **smoking** might have the most significant impact on predicting diabetes.
    
    In this case, the predictor variables are age, smoking, and alcohol consumption, while the target variable is smoking.
    
- Higher-level nodes are selected to have predictor variables that best separate the target variable.

For example, guessing a number between 1 and 100 can be done by finding the median value.

In statistics, since probability is considered, choosing the median value is wise. This approach is an example of a decision tree model.

→ It is a model that splits decisions into two branches (binary tree structure).

### Building a Decision Tree Model

1) Preprocessing

2) Model Creation

3) Prediction and Performance Evaluation

→ These are the stages involved.

## Cross-Validation

Cross-validation involves splitting the data into parts where some are used to build the model and others are used to evaluate the model’s performance.

**Training Set**: Data used to build the model.

**Test Set**: Data used to evaluate the model's performance.

Using the `scikit-learn` package, you can split the data into training and test sets.

- test_size: The proportion of the test set ($0 < p < 1$).
- stratify: The variable for which class proportions should be maintained.
- random_state: The random seed value.

![cross](https://i.imgur.com/PDsSfph.png)

**Interpreting Decision Tree Nodes**

1. The proportion of observations that fall into a node.
(This shows what percentage of the entire dataset falls into that particular node.)
2. The class proportions of the target variable.
3. The predominant class based on a 0.5 threshold (which of the two target variable classes is more prevalent).
4. The splitting criterion: The basis for splitting the node → One-hot encoding: Numerical encoding where a specific value is 1, and others are 0.

![Confusion Matrix](https://i.imgur.com/B0y8RBG.png)

True/False: Whether the prediction was correct, Positive/Negative: The predicted direction.

First Column

→ The model predicted 2,382 people (1,801+582) as high income; actually, 1,901 were high income, and 582 were low income.

Second Column

→ The model predicted 12,270 people (1,705+10,565) as low income; actually, 1,705 were high income, and 10,565 were low income.

Accuracy (the proportion of correct predictions): $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$

Precision (the proportion of positive predictions that are correct): $\frac{\text{TP}}{\text{TP} + \text{FP}}$

Recall (the proportion of actual positives correctly identified): $\frac{\text{TP}}{\text{TP} + \text{FN}}$

F1 Score: Reflects both precision and recall.

```python
## decisiontree.py

def p(str):
    print(str, '\n')

## Creating an Income Prediction Model
import pandas as pd
df = pd.read_csv('./assets/adult.csv')
df.info()
# In the csv file, income is the dependent variable, and the rest are independent variables.
p('')

# Data Preprocessing
import numpy as np

# Label income as 'high' if it exceeds 50K, otherwise 'low'
df['income'] = np.where(df['income'] == '>50K', 'high', 'low')
p(df['income'].value_counts(normalize=True)) # normalize=True to show category proportions

# Remove unnecessary variables that don't help predict the target variable
df = df.drop(columns='fnlwgt')

# Convert string variables to numeric types using One-Hot Encoding
# One-hot encoding: representing data as 1s and 0s

target = df['income']
df = df.drop(columns='income') # Drop the target variable
df = pd.get_dummies(df)
df['income'] = target
df.info(max_cols=np.inf) # Display information regardless of the number of variables

# Split the dataset into training and test sets using scikit-learn
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df,
                                     test_size=0.3,             # Test set proportion
                                     stratify=df["income"],     # Variable to maintain category proportions
                                     random_state=1234)         # Random seed value
p(df_train.shape)   # Number of rows and variables in the training set
p(df_test.shape)    # Number of rows and variables in the test set

# Category proportions in the training and test sets
p(df_train["income"].value_counts(normalize=True))
p(df_test["income"].value_counts(normalize=True))

## Creating a Decision Tree Model

# Import libraries
from sklearn import tree

# Create a decision tree classifier
# random_state: random seed value, max_depth: maximum depth of the tree
clf = tree.DecisionTreeClassifier(random_state=1234, max_depth=3)

# Extract predictor and target variables
train_x = df_train.drop(columns='income') # Predictor variables (all variables except the target variable)
train_y = df_train['income'] # Target variable (the variable to be predicted using the predictor variables)

# Create the decision tree model
# fit(X: Predictor variables, y: Target variable)
model = clf.fit(X=train_x, y=train_y) # Each predictor and target variable

# Visualize the model
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        'figure.dpi': 100, # Graph resolution
        'figure.figsize': [12, 8] # Graph size
    }
)

# Tree graph
tree.plot_tree(
    model, # Model
    feature_names=train_x.columns, # Predictor variable names
    class_names=['high', 'low'], # Target variable classes (alphabetical order)
    proportion=True, # Show proportions
    filled=True, # Fill nodes with colors
    rounded=True, # Round the corners of nodes
    impurity=False, # Hide impurity values
    label='root', # Position of labels
    fontsize=10 # Font size
)
plt.show()
# Experiment with different options

# Extract predictor and target variables for prediction
test_x = df_test.drop(columns='income') # Predictor variables
test_y = df_test['income'] # Target variable

# Predict using the model
df_test['pred'] = model.predict(test_x)
p(df_test)

## Performance Evaluation

# Create a confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(
    y_true=df_test['income'], # Actual values
    y_pred=df_test['pred'], # Predicted values
    labels=['high', 'low'] # Labels (order of classes, alphabetical order)
)
p(conf_matrix)

## Display the confusion matrix as a heatmap

# Reset graph settings
plt.rcdefaults() # Reset graph settings
from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, # Matrix data
    display_labels=('high', 'low') # Target variable class names
)

p.plot(cmap='Blues') # Colormap
plt.show()

## Calculate Performance Metrics

import sklearn.metrics as metrics

# Accuracy
acc = metrics.accuracy_score(
    y_true=df_test['income'], # Actual values
    y_pred=df_test['pred'] # Predicted values
)
print(acc)
# 0.8439227461953184

# Precision
pre = metrics.precision_score(
    y_true=df_test['income'], # Actual values
    y_pred=df_test['pred'], # Predicted values
    pos_label='high' # Target class
)
print(pre)
# 0.7557700377675199

# Recall
rec = metrics.recall_score(
    y_true=df_test['income'], # Actual values
    y_pred=df_test['pred'], # Predicted values
    pos_label='high' # Target class
)
print(rec)
# 0.5136908157444381

# F1 Score
f1 = metrics.f1_score(
    y_true=df_test['income'], # Actual values
    y_pred=df_test['pred'], # Predicted values
    pos_label='high' # Target class
)
print(f1)
# 0.6116488368143997
# The F1 score reflects both recall and precision and is about midway between them, though not an average.
```

```python
# decisionmodellib.py

## Decision Tree Model Library

# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Function to load dataset
def importdata(filename):
    balance_data = pd.read_csv(
        filename,
        sep=',',
        header=None
    )
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset (Top 5 Rows): ", balance_data.head())
    return balance_data

# print(importdata(filename))

# Function to split data into predictor and target variables
def splitdataset(balance_data, X, Y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X, Y, X_train, X_test, y_train, y_test

# Function to create and train the decision tree classifier
def train_using(criterion, random_state, md, msl, X_train, X_test, y_train):
    clf = DecisionTreeClassifier(
        criterion=criterion,  # Type of classifier (algorithm)
        random_state=random_state,  # Random seed value
        max_depth=md,  # Tree depth
        min_samples_leaf=msl  # Minimum leaf node size
    )
    clf.fit(X_train, y_train)
    return clf

# Function to make predictions
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values: ", y_pred)
    return y_pred

# Function to display confusion matrix, accuracy, and report
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
    print("Report: ", classification_report(y_test, y_pred))

# Function to visualize decision tree
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

# Execution
filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
df = importdata(filename)
X = df.values[:, 1:5]  # Predictor variables
Y = df.values[:, 0]  # Target variables
X, Y, X_train, X_test, y_train, y_test \
    = splitdataset(df, X, Y, 0.3, 1234)
clf = train_using('gini', 1234, 3, 5, X_train, X_test, y_train)
plot_decision_tree(
    (15, 10),
    clf,
    ['X1', 'X2', 'X3', 'X4'],
    ['L', 'B', 'R']
)
```

# Korean

## 머신 러닝 모델

머신러닝 모델은 현재까지의 데이터로 미래의 데이터를 예측하기 위한 모델이다.

- 예측변수
    
    → 예측하는 데 활용하는 변수 또는 모델에 입력하는 값
    
- 타겟변수
    
    → 예측하고자 하는 변수 또는 모델이 출력하는 값
    
**예측 변수**를 통해 **타겟 변수**를 찾아내기 위한 모델.

$y = f(x)$에서, $y$가 타겟변수이며, $f(x)$가 예측변수이다.

## 의사 결정 나무 모델

- **예/아니오**로 2분된 답변을 연속적으로 취득하여 예측하는 모델.
- 구조가 단순하고 이해하기 쉬워 많은 예측 모델의 토대가 됨.
- 타겟 변수를 가장 잘 분리해주는 예측 변수에 주목하는 모델.

    예시) 나이, 흡연, 음주의 예측 변수 중 당뇨병에 가장 많은 영향을 미치는 것은 **흡연**이다.
    
    결국 예측 변수는 나이, 흡연, 음주이며
    
    타겟변수는 흡연이다.
    
- 상위 노드일수록 타겟 변수를 잘 분리해주는 예측 변수를 선택하도록 함.

예시) 1~100 사이 숫자를 맞추는 경우 → 중간 값을 찾는 것.

통계학에서는 확률 기반이므로 중간 값을 채택하는 것이 현명하다. 이와 같은 경우 의사 결정 나무 모델을 이용하는 것이다.

→ 이분화하는 모델을 의미한다. (이진 트리 구조)

### 의사결정나무 모델 만들기

1) 전처리

2) 모델 생성

3) 예측 및 성능 평가

→ 이러한 단계로 구성된다.

## 교차 검증

데이터를 분할해 일부는 모델 생성에 사용, 나머지는 성능 평가에 사용.

**트레이닝 세트**: 모델 생성에 사용되는 데이터.

**테스트 세트**: 성능을 평가하는 데 사용되는 데이터.

`scikit-learn` 패키지를 이용해 트레이닝 세트와 테스트 세트로 분할.

- test_size: 테스트 세트의 비율 ($0 < p < 1$)
- stratify: 범주별 비율을 통일할 변수
- random_state: 난수 초기값

![cross](https://i.imgur.com/PDsSfph.png)

**의사결정나무 노드 해석**

1. 노드에 해당하는 관측치 비율
   (전체 데이터의 몇 퍼센트가 해당 노드로 분류되었는지)
2. 타겟 변수의 클래스별 비율
3. 0.5 기준의 우세한 클래스 (타겟 변수의 두 클래스 중 어느 쪽이 더 많은지)
4. 분리 기준: 노드의 분리 기준 → 원핫인코딩: 특정값이면 1, 아니면 0으로 숫자화!

![Confusion Matrix](https://i.imgur.com/B0y8RBG.png)

정답 여부: True/False, 예측 방향: Positive/Negative

첫 번째 열

→ 모델이 2,382명 (1,801+582)을 high로 예측, 실제 high는 1,901, low는 582.

두 번째 열

→ 모델이 12,270명(1,705+10,565)을 low로 예측, 실제 high는 1,705, low는 10,565.

정확도(예측해서 맞춘 비율): $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$

정밀도(관심 클래스를 예측해서 맞춘 비율): $\frac{\text{TP}}{\text{TP} + \text{FP}}$

재현율(실제 데이터에서 관심 클래스를 찾아낸 비율): $\frac{\text{TP}}{\text{TP} + \text{FN}}$

F1 Score: 정밀도와 재현율을 함께 반영.

```python
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

# 연소득 5만 달러 초과하면 high, 그렇지 않으면 low로
df['income'] = np.where(df['income'] == '>50K', 'high', 'low')
p(df['income'].value_counts(normalize=True)) # normalize=True 범주 비율로 출력하는 방법

# 타겟 변수(종속 변수) 예측에 도움이 안 되는 불필요한 변수 제거
df = df.drop(columns='fnlwgt')

# 원핫인코딩(one-hot encoding)을 이용한 문자 타입 변수를 숫자 타입으로 변환
# 원핫 인코딩: 데이터를 모두 1이나 0으로 표시

target = df['income']
df = df.drop(columns='income') # 종속 변수 제거
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

# 의사결정나무 분류기 생성
# random_state: 랜덤 시드값, max_depth: 트리의 최대 깊이
clf = tree.DecisionTreeClassifier(random_state=1234, max_depth=3)

# 예측 변수, 타겟 변수 추출
train_x = df_train.drop(columns='income') # 예측 변수 (전체 15개 변수 중 타겟 변수를 제외한 것)
train_y = df_train['income'] # 타겟 변수 (예측 변수를 통해 예측하고자 하는 변수)

# 의사결정 나무 모델 만들기
# fit(X : 예측 변수, y : 타겟 변수)
model = clf.fit(X=train_x, y=train_y) # 각각 예측 변수와 타겟 변수

# 모델 시각화
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        'figure.dpi': 100, # 그래프 해상도
        'figure.figsize': [12, 8] # 그래프 크기
    }
)

# 트리 그래프
tree.plot_tree(
    model, # 모델
    feature_names=train_x.columns, # 예측 변수의 명칭들
    class_names=['high', 'low'], # 타겟 변수 클래스(알파벳 오름순)
    proportion=True, # 비율 표시 여부
    filled=True, # 채우기 여부
    rounded=True, # 노드 테두리를 둥글게 할지 여부
    impurity=False, # 불순도 표시 여부
    label='root', # 제목 표시 위치
    fontsize=10 # 글자 크기
)
plt.show()
# 옵션들을 바꾸며 그려보기

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
    y_true=df_test['income'], # 실제 값
    y_pred=df_test['pred'], # 예측 값
    labels=['high', 'low'] # 레이블 (클래스 배치 순서, 문자 오름차순)
)
p(conf_matrix)

## 컨퓨전 메트릭스를 히트맵으로 표시

# 그래프 설정 되돌리기
plt.rcdefaults() # 그래프 설정 초기화
from sklearn.metrics import ConfusionMatrixDisplay

p = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, # 메트릭스 데이터
    display_labels=('high', 'low') # 타겟 변수 클래스명
)

p.plot(cmap='Blues') # 컬러맵
plt.show()

## 성능평가 지표 구하기

import sklearn.metrics as metrics

# Accuracy : 정확도
acc = metrics.accuracy_score(
    y_true=df_test['income'], # 실제 값
    y_pred=df_test['pred'] # 예측 값
)
print(acc)
# 0.8439227461953184

# Precision : 정밀도
pre = metrics.precision_score(
    y_true=df_test['income'], # 실제 값
    y_pred=df_test['pred'], # 예측 값
    pos_label='high' # 관심 클래스
)
print(pre)
# 0.7557700377675199

# Recall : 재현율
rec = metrics.recall_score(
    y_true=df_test['income'], # 실제 값
    y_pred=df_test['pred'], # 예측 값
    pos_label='high' # 관심 클래스
)
print(rec)
# 0.5136908157444381

# F1 Score
f1 = metrics.f1_score(
    y_true=df_test['income'], # 실제 값
    y_pred=df_test['pred'], # 예측 값
    pos_label='high' # 관심 클래스
)
print(f1)
# 0.6116488368143997
# f1 값은 재현율과 정밀도 중간 정도로 나온다. 평균은 아니다.
```

```python
# decisionmodellib.py

## 디시젼트리 모델 라이브러리

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
    print("데이터셋 길이: ", len(balance_data))
    print("데이터셋 모양: ", balance_data.shape)
    print("데이터셋 (상위 5행): ", balance_data.head())
    return balance_data

# print(importdata(filename))

# 예측변수, 타겟변수 분리 함수
def splitdataset(balance_data, X, Y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X, Y, X_train, X_test, y_train, y_test

# 디시젼트리 분류기 생성 및 학습 함수
def train_using(criterion, random_state, md, msl, X_train, X_test, y_train):
    clf = DecisionTreeClassifier(
        criterion=criterion,  # 분류기의 종류(분류 알고리즘)
        random_state=random_state,  # 랜덤 시드값
        max_depth=md,  # 트리 깊이
        min_samples_leaf=msl  # 리프 노드 최소 개수
    )
    clf.fit(X_train, y_train)
    return clf

# 예측 함수
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("예측값: ", y_pred)
    return y_pred

# 컨퓨전 메트릭스, 정확도, 레포트 출력 함수
def cal_accuracy(y_test, y_pred):
    print("컨퓨전 메트릭스: ", confusion_matrix(y_test, y_pred))
    print("정확도: ", accuracy_score(y_test, y_pred) * 100)
    print("레포트: ", classification_report(y_test, y_pred))

# 디시젼트리 가시화 함수
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
```











