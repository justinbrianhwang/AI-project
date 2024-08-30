# English

```markdown
## Cross-Validation

1. **Cross-Validation**
    
    Cross-validation involves splitting the training dataset into a training set, validation set, and test set. The validation set is used by iteratively taking subsets of the training set to validate the model, thereby increasing the accuracy of the validation.
    
    **Training Dataset (Train)**: Data used for training the model.
    
    **Validation Dataset (Validation)**: A subset of the training dataset used for the preliminary evaluation of the model's performance.
    
    **Test Dataset (Test)**: Data used for final performance evaluation after all training and validation processes are complete.
    
    **Advantages of Cross-Validation**: 
    - All datasets can be used for training.
    - Reduces bias by preventing evaluation on skewed data subsets.
    - Leads to more generalized model performance.
    
    **Disadvantages of Cross-Validation**:
    - Increases model training/evaluation time due to the repeated process.
    - Overall evaluation performance may decrease with more cross-validation iterations.
    
    1. **$k$-Fold Cross-Validation**
        
        $k$-Fold Cross-Validation is a method where $k$ iterations of training and validation are performed on $k$ subsets (folds) of the data. This method allows you to set the number of cross-validations.
        
        For example, $k=5$ (five validations):
        
        Fold1   Fold2   Fold3   Fold4   Fold5 
        
        Validation     Train     Train      Train      Train     
        
        Train     Validation      Train      Train      Train 
        
        Train     Train      Validation      Train      Train 
        
        Train     Train      Train      Validation      Train 
        
        Train     Train      Train      Train      Validation 
        
        → This process illustrates how cross-validation is performed.
        
        ### Principles
        
        1. Divide the training set into $k$ folds (subsets of the training set).
        2. Use the first fold as the validation set and the remaining $k-1$ folds as the training set.
        3. Train the model and evaluate it on the validation set.
        4. Repeat the process, using the next fold as the validation set each time.
        
        ### Process
        
        1. Data Preparation: Split into training and test datasets.
        2. Data Splitting: Divide the data into $k$ folds.
        3. Model Training and Evaluation: Use one fold as the test set and the remaining $k-1$ folds as the training set.
        4. Repeat $k$ times: Repeat steps 1-3 for $k$ times, ensuring that each fold is used as a test set once.
        5. Performance Evaluation: Average the performance metrics obtained from the $k$ evaluations to assess the model's overall performance.
        
        ### Problems with $k$-Fold Cross-Validation
        
        Since the data is split randomly, there's a risk that the **class ratio** in each fold might differ from the **class ratio** in the overall dataset. 
        
        → For example, if you want to predict the average 100m running time for a class, but the distribution is skewed, the performance in different folds might vary, leading to biased evaluation results. It's important to use data with evenly distributed variance.
        
2. **Stratified $k$-Fold**
    
    Stratified $k$-Fold addresses the issues of standard $k$-Fold Cross-Validation by ensuring that the class distribution is balanced across all folds.
    
    This technique is used in classification tasks to maintain a class distribution in each fold that is similar to the original dataset's distribution.
    
    This results in a more accurate assessment of the model's performance.
    
    → It uses a balanced classification method before performing cross-validation.
    
    ### Process
    
    1. Class Distribution Check: Check the class distribution of the dataset.
                For classification tasks, the class labels are known, and the number of samples per class is calculated.
    
    2. Fold Creation: Divide the dataset into $k$ folds.
                 Each fold should maintain a class distribution similar to the original dataset.
    
    3. Model Training and Evaluation:
                 Use one of the $k$ folds as the test set, and the remaining $k-1$ folds as the training set.
    
    4. Repeat $k$ times: Repeat steps 1-3.
    

1. **Cross_val_score() Function**
    - A function that facilitates cross-validation.
    - Provided by the scikit-learn library, this function is used to perform cross-validation and evaluate the model's performance.
    - It accepts parameters such as the specified model, dataset, and cross-validation method, and returns the accuracy of the cross-validation.
    - Parameters:
        - estimator: A classifier or regressor algorithm.
        - features: Feature dataset.
        - label: Label dataset.
        - scoring: Prediction performance evaluation metric.
            
                Classification ⇒ Accuracy, Precision, Recall, F1-Score.
            
                 Regression ⇒ Mean Squared Error (MSE), R2 Score.
            
        - cv: Number of folds.

```python
# titanicdf.py

def p(str):
    print(str, "\n")

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings # Library for handling warnings
warnings.filterwarnings(action="ignore") # Ignore warning messages

# Load train.csv
titanicDf = pd.read_csv('./assets/train.csv')

# Show the first 5 rows
p(titanicDf.head())

# Information about the DataFrame -> Always check this information first
titanicDf.info()

# Explanation of train.csv columns
"""
PassengerId : Serial number of passenger data
Survived : Survival status (0: deceased, 1: survived)
Pclass : Class (1: First class, 2: Second class, 3: Third class)
Name : Name
Sex : Gender
Age : Passenger age
SibSp : Number of siblings/spouses aboard
Parch : Number of parents/children aboard
Ticket : Ticket number
Fare : Fare
Cabin : Cabin number
Embarked : Port of embarkation (C: Cherbourg, Q: Queenstown, S: Southampton)  
"""

## Data preprocessing -> 80~90% of the work.
# Check how many null values each column has
p(titanicDf.isnull().sum())

# Fill Age null values with the mean
titanicDf['Age'].fillna(titanicDf['Age'].mean(), inplace=True) # Preprocessing method
p(titanicDf.isnull().sum())

# Fill Cabin and Embarked null values with 'N'
titanicDf['Cabin'].fillna('N', inplace=True)
titanicDf['Embarked'].fillna('N', inplace=True)

p(titanicDf.isnull().sum())

# Check for duplicates
p(titanicDf[titanicDf.duplicated()])

# Number of passengers by gender
p(titanicDf['Sex'].value_counts())

# Number of passengers by cabin
p(titanicDf['Cabin'].value_counts())

# Number of passengers by port of embarkation
p(titanicDf['Embarked'].value_counts())

# Extract the first letter of the cabin information and check the count
titanicDf['Cabin'] = titanicDf['Cabin'].str[:1]
p(titanicDf['Cabin'].value_counts())

# Check the number of survivors by gender
p(titanicDf['Survived'].value_counts())
p(titanicDf['Sex'].value_counts())

# Grouping
p(titanicDf.groupby('Sex')['Survived'].sum())

# Plot the number of survivors by gender
ss = titanicDf.groupby(['Sex', 'Survived']).size().unstack()
ss.plot(kind='bar')
plt.show()

# Seaborn plot
# estimator=len: Counts the size of each group
sns.barplot(x='Sex', y='Survived', data=titanicDf, hue='Sex', estimator=len)
plt.show()

# Create age categories
def category(age):
    re = ''
    if age <= -1:
        re = 'Unknown'
    elif age <= 5:
        re = 'baby'
    elif age <= 12:
        re = 'child'
    elif age <= 19:
        re = 'teenager'
    elif age <= 25:
        re = 'student'
    elif age <= 35:
        re = 'young adult'
    elif age <= 80:
        re = 'adult'
    else :
        re = 'elderly'
    return re

# Plot age categories

# Set graph size
plt.figure(figsize=(10, 6))

# Set X-axis values in order
group_name = ["Unknown", "baby", "child", "teenager", "student",
              "young adult", "adult", "elderly"]

# Add the age category to the DataFrame
titanicDf['Age_Cate'] = titanicDf['Age'].apply(category)
p(titanicDf.head())

sns.barplot(
    x='Age_Cate',
    y='Survived',
    hue='Sex',
    data=titanicDf,
    order=group_name
)
plt.show()

# Plot the number of survivors by family size
titanicDf['Family'] = titanicDf['SibSp'] +  titanicDf['Parch'] + 1
sns.barplot(
    data=titanicDf,
    x='Family',
    y='Survived',
    estimator=len
)
plt.show()

# Plot the fare distribution
sns.histplot(
    data=titanicDf,
    x='Fare',
    bins=30, # Bin size
    kde=True # Kernel Density Estimation
)
plt.show()

# Compare age and survival rate (Histogram, Boxplot)
sns.histplot(data=titanicDf, x='Age', hue='Survived', bins=30, kde=True)
plt.show()
sns.boxplot(data=titanicDf, x='Survived', y='Age')
plt.show()
```

## Encoding

In **machine learning**, encoding refers to the process of converting string values into numerical values for computation.

1. **Label Encoding**
    - Converts string values by sorting them in descending order and then assigning incremental numerical values starting from 0.
    
    For example:
    
    Hong          Lee           0
    Kang    ⇒    Kang    ⇒    1
    Lee           Kim           2
    Kim           Hong          3
    
    - Label encoding is applied to tree-based models where the numerical difference does not affect the model (e.g., decision trees, random forests).
    
2. **One-Hot Encoding**
    - Converts categorical data into binary values (0, 1) for each category.
    - Often used in linear models where each category is treated independently.
    

```python
## encdoing.py

def p(str):
    print(str, '\n')

# Import libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

# Load DataFrame
df = pd.read_csv('./assets/train.csv')

# Label Encoding
from sklearn.preprocessing import LabelEncoder # Preprocessing library

# Create label encoder
label_encoder = LabelEncoder()

# Encode gender using label encoding
df['Sex_LableEncoder'] = label_encoder.fit_transform(df['Sex']) # Apply the rule in the function
p(label_encoder.classes_) #
p(df.head())

# Function for label encoding
def encode_features(df, features): # For general use, pass features as an argument
    for i  in features:
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])
    return df # Returns the DataFrame with features label encoded

features = ['Sex', 'Cabin', 'Embarked']
p(encode_features(df, features)) # Confirm successful transformation
```

```python
## crossvaildation.py

## Cross-validation model training and performance evaluation

def p(str):
    print(str, '\n')

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

# Create DataFrame
df = pd.read_csv('./assets/train.csv')
#p(df.info())

# Split into features and labels
features = df.drop('Survived', axis=1) # Features
labels = df['Survived'] # Labels
# p(features)
# p(labels)

# Preprocess features data
# Remove missing values and unnecessary attributes (Passenger Id, Name, Ticket)

# Check for null values
p(features.isnull().sum())

# Handle missing values
features["Age"].fillna(features["Age"].mean(), inplace=True)
features['Cabin'].fillna('N', inplace=True)
features['Embarked'].fillna('N', inplace=True)

# Remove unnecessary attributes (PassengerId, Name, Ticket)
features = features.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
#p(features) # Remove unnecessary information

# Perform label encoding -> Necessary for numerical conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['Sex'] = le.fit_transform(features['Sex'])
features['Embarked'] = le.fit_transform(features['Embarked'])
features['Cabin'] = le.fit_transform(features['Cabin'])
# p(features) # Convert all values to numbers for machine learning

# Split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, # feature
    labels, # lable
    test_size=0.2, # Split ratio (20% test set, 80% train set)
    random_state=53 # Random seed value
)

# Check Shape -> Confirm sizes
# p(X_train.shape) # (712, 8)
# p(X_test.shape) # (179, 8)
# p(y_train.shape) # (712,)
# p(y_test.shape) # (179,)

## Model training

# Decision Tree -> Binary splitting method
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=11)
# p(dt)

# Train the model
dt.fit(X_train, y_train)

# Predict
pred = dt.predict(X_test)

# Accuracy -> A commonly used function to remember
p(f"Accuracy: {accuracy_score(y_test, pred)}")

## Cross-validation

# Load library
from sklearn.model_selection import cross_val_score

# dt: DataFrame, features: features, labels: labels, cv: number of folds
scores = cross_val_score(dt, features, labels, cv=10)

# Calculate accuracy for each cross-validation
# iter_count: iteration count, accuracy: accuracy
# enumerate(scores): Enumerate cross-validation results
for iter_count, accuracy in enumerate(scores):
    p(f"{iter_count + 1}th cross-validation accuracy: {accuracy}")

# Mean accuracy
p(np.mean(scores))

# 10-fold : 0.7811860174781523
# 20-fold : 0.7756565656565656
# 50-fold : 0.7949673202614378
# 100-fold : 0.7861111111111111 
# As observed, increasing the number of folds does not necessarily improve accuracy.
```



# Korean

~~~markdown
## 교차 검증

1. **교차 검증**
    
    교차 검증은 학습 데이터 셋을 학습, 검증, 평가 데이터 셋으로 분리하는 방법이다. 검증 세트를 학습 세트의 일부로 교차하여 사용함으로써, 검증의 정확도를 높일 수 있다.
    
    **학습 데이터 셋 (Train)**: 모델을 학습시키기 위한 데이터.
    
    **검증 데이터 셋 (Validation)**: 학습 데이터 셋의 일부를 추출하여 모델의 성능을 평가하는데 사용.
    
    **평가 데이터 셋 (Test)**: 학습과 검증이 완료된 후 최종적으로 성능 평가에 사용되는 데이터.
    
    **교차 검증의 장점**: 
    - 모든 데이터를 학습에 활용할 수 있다.
    - 평가 데이터의 편중을 방지할 수 있다.
    - 일반화된 모델 성능을 확보할 수 있다.
    
    **교차 검증의 단점**:
    - 반복 횟수가 많아질수록 모델 훈련/평가 시간이 길어진다.
    - 교차 검증 횟수가 많아질수록 전체적인 평가 성능이 저하될 수 있다.
    
    1. **$k$-폴드 교차 검증 ($k$-Fold Cross-Validation)**
        
        $k$-폴드 교차 검증은 데이터를 $k$개의 폴드로 나누어 $k$ 번의 학습과 검증을 반복 수행하는 방법이다.
        
        예시: $k=5$ (5번 검증)
        
        폴드1   폴드2   폴드3   폴드4   폴드5 
        
        검증     학습     학습      학습      학습     
        
        학습     검증      학습      학습      학습 
        
        학습     학습      검증      학습      학습 
        
        학습     학습      학습      검증      학습 
        
        학습     학습      학습      학습      검증 
        
        → 이러한 방식으로 교차 검증을 수행.
        
        ### 원리
        
        1. 훈련 세트를 $k$개의 폴드로 나눈다.
        2. 첫 번째 폴드를 검증 세트로 사용하고 나머지 $k-1$개의 폴드를 학습 세트로 사용.
        3. 모델을 훈련하고 검증 세트로 평가.
        4. 차례대로 다음 폴드를 검증 세트로 사용하며 반복.
        
        ### 과정
        
        1단계) 데이터 준비: 학습 데이터와 테스트 데이터로 나눈다.
        
        2단계) 데이터 분할: 데이터를 $k$개의 폴드로 나눈다.
        
        3단계) 모델 학습과 평가: $k$개의 폴드 중 하나를 테스트 세트로 사용하고 나머지 $k-1$개의 폴드를 학습 세트로 사용.
        
        4단계) $k$번 반복: 1~3 단계를 $k$번 반복하여 각 폴드가 한 번씩 테스트 세트로 사용되도록 한다.
        
        5단계) 성능 평가: $k$번의 평가를 통해 얻은 성능 지표를 평균 내어 최종적인 모델 성능을 평가.
        
        ### $k$-폴드 교차 검증의 문제점
        
        무작위로 데이터를 분할하기 때문에 각 폴드에서 **클래스 비율**이 전체 데이터 셋의 **클래스 비율**과 다를 수 있다.
        
        → 예를 들어, 반 학생의 100m 달리기 평균을 예측할 때, 반마다 편중이 발생할 수 있다. 이를 방지하기 위해 분산이 고르게 분포된 데이터를 사용하는 것이 중요하다.
        
2. **Stratified $k$-Fold**
    
    Stratified $k$-Fold는 기존 $k$-폴드 교차 검증의 문제점을 해결하기 위해 사용된다.
    
    분류 문제에서 클래스 별로 균형을 유지하면서 데이터를 분할하는 기법이다.
    
    각 폴드에서 클래스 비율이 원래 데이터 셋과 유사하도록 보장하여 모델의 성능 평가를 더 정확하게 수행할 수 있다.
    
    → 균등하게 분류한 뒤 사용하는 방법이다.
    
    ### 과정
    
    1단계) 클래스 분포 확인: 데이터 셋의 클래스 분포를 확인.
    
    2단계) 폴드 생성: 데이터 셋을 $k$개의 폴드로 나누며, 각 폴드에서 클래스 비율이 원래 데이터 셋과 유사하게 유지되도록 한다.
    
    3단계) 모델 학습과 평가: $k$개의 폴드 중 하나를 테스트 세트로 사용하고 나머지 $k-1$개의 폴드를 학습 세트로 사용.
    
    4단계) $k$번 반복: 1~3단계 과정을 반복.
    

1. **Cross_val_score() 함수**
    - 교차 검증을 간편하게 수행할 수 있는 함수.
    - 사이킷런 라이브러리에서 제공하며, 모델의 성능을 평가하는데 사용된다.
    - 지정된 모델과 데이터 셋, 교차 검증 방법을 매개변수로 받아 정확도를 반환한다.
    - 매개변수:
        - estimator: 분류 알고리즘(Classifier) 또는 회귀 알고리즘(Regressor).
        - features: 특성 데이터 셋.
        - label: 레이블 데이터 셋.
        - scoring: 예측 성능 평가 지표.
            
                분류 ⇒ 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score.
            
                회귀 ⇒ 평균제곱오차(MSE), 결정 계수(R2 Score).
            
        - cv: 폴드 수.

~~~python
# titanicdf.py

def p(str):
    print(str, "\n")

# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings # 경고 메시지를 다루기 위한 라이브러리
warnings.filterwarnings(action="ignore") # 경고 메시지 무시

# train.csv를 가져옴
titanicDf = pd.read_csv('./assets/train.csv')

# 상위 5개의 행 출력
p(titanicDf.head())

# 데이터 프레임 정보 -> 항상 기본적으로 확인해야 할 정보
titanicDf.info()

# train.csv 컬럼 설명
"""
PassengerId : 탑승자 데이터 일련번호
Survived : 생존 여부 (0: 사망, 1: 생존)
Pclass : 선실등급(1: 일등석, 2: 이등석, 3: 삼등석)
Name : 이름
Sex : 성별
Age : 탑승자 나이
SibSp : 같이 탑승한 형제, 자매, 또는 배우자의 인원 수
Parch : 같이 탑승한 부모님 또는 자녀의 인원 수
Ticket : 티켓 번호 
Fare: 요금
Cabin: 선실 번호 
Embarked: 중간 정착한 항구 (C: Cherbourg, Q: Queenstown, S: Southampton)  
"""

## 데이터 전처리 -> 80~90%의 작업.
# 컬럼별로 null 값이 몇 개 있는지 확인
p(titanicDf.isnull().sum())

# 나이 null을 평균값으로 채우기
titanicDf['Age'].fillna(titanicDf['Age'].mean(), inplace=True) # 전처리 방법
p(titanicDf.isnull().sum())

# 선실과 정착지를 'N'으로 채우기
titanicDf['Cabin'].fillna('N', inplace=True)
titanicDf['Embarked'].fillna('N', inplace=True)

p(titanicDf.isnull().sum())

# 중복값 확인
p(titanicDf[titanicDf.duplicated()])

# 성별에 따른 승객 수
p(titanicDf['Sex'].value_counts())

# 선실 별 승객 수
p(titanicDf['Cabin'].value_counts())

# 정착지별 승객 수 확인
p(titanicDf['Embarked'].value_counts())

# 선실 정보의 첫 번째 알파벳만 추출 후 개수 확인
titanicDf['Cabin'] = titanicDf['Cabin'].str[:1]
p(titanicDf['Cabin'].value_counts())

# 성별에 따른 생존자 수 확인
p(titanicDf['Survived'].value_counts())
p(titanicDf['Sex'].value_counts())

# 그룹핑
p(titanicDf.groupby('Sex')['Survived'].sum())

# 성별과 생존자 수를 기준으로 그래프 그리기
ss = titanicDf.groupby(['Sex', 'Survived']).size().unstack()
ss.plot(kind='bar')
plt.show()

# seaborn 그래프
# estimator=len: 각 그룹의 크기를 계산해서 사용
sns.barplot(x='Sex', y='Survived', data=titanicDf, hue='Sex', estimator=len)
plt.show()

# 나이에 따른 카테고리 생성
def category(age):
    re = ''
    if age <= -1:
        re = 'Unknown'
    elif age <= 5:
        re = 'baby'
    elif age <= 12:
        re = 'child'
    elif age <= 19:
        re = 'teenager'
    elif age <= 25:
        re = 'student'
    elif age <= 35:
        re = 'young adult'
    elif age <= 80:
        re = 'adult'
    else:
        re = 'elderly'
    return re

# 나이 카테고리 그래프 그리기

# 그래프 사이즈 정하기
plt.figure(figsize=(10, 6))

# X축 값을 순차적으로 표시
group_name = ["Unknown", "baby", "child", "teenager", "student",
              "young adult", "adult", "elderly"]

# 나이에 따른 카테고리 -> DataFrame에 추가
titanicDf['Age_Cate'] = titanicDf['Age'].apply(category)
p(titanicDf.head())

sns.barplot(
    x='Age_Cate',
    y='Survived',
    hue='Sex',
    data=titanicDf,
    order=group_name
)
plt.show()

# 가족 또는 동승자 수와 생존 여부에 따른 막대 그래프
titanicDf['Family'] = titanicDf['SibSp'] + titanicDf['Parch'] + 1
sns.barplot(
    data=titanicDf,
    x='Family',
    y='Survived',
    estimator=len
)
plt.show()

# 탑승 요금 분포 그래프
sns.histplot(
    data=titanicDf,
    x='Fare',
    bins=30, # 구간 분할
    kde=True # Kernel Density Estimation 부드러운 곡선으로 표시
)
plt.show()

# 나이 따른 생존율 비교 (히스토그램, 상자 그림)
sns.histplot(data=titanicDf, x='Age', hue='Survived', bins=30, kde=True)
plt.show()
sns.boxplot(data=titanicDf, x='Survived', y='Age')
plt.show()
~~~

## 인코딩 (Encoding)

**머신 러닝**에서 데이터를 연산시키기 위해서 문자열 값들을 숫자화하는 과정을 말한다.

1. **레이블 인코딩** 
    - 문자열 값들을 내림차순으로 정렬한 후 0부터 1씩 증가하는 값으로 변환하는 것이다.
    
    예시:
    
    Hong          Lee           0
    Kang    ⇒    Kang    ⇒    1
    Lee           Kim           2
    Kim           Hong          3
    
    - 레이블 인코딩은 숫자의 차이가 모델에 영향을 주지 않는 트리 계열의 모델에 적용 
    (의사결정나무, 랜덤 포레스트 등)
    
2. **원 핫 인코딩 (One-Hot Encoding)**
    - 대상 데이터를 목록화해서 그 목록에 대한 이진값(0, 1)으로 변환하는 인코딩. 쉽게 말해서, 조건에 만족하면 1, 그렇지 않으면 0으로 변환.
    - 범주형 변수의 각 범주가 서로 독립적으로 다루어지는 선형 모델에서 자주 사용됨.

~~~python
## encdoing.py

def p(str):
    print(str, '\n')

# 라이브러리 임포트
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

# 데이터 프레임 로딩
df = pd.read_csv('./assets/train.csv')

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder # 전처리 라이브러리

# 레이블 인코더 생성
label_encoder = LabelEncoder()

# 성별을 레이블 인코딩
df['Sex_LableEncoder'] = label_encoder.fit_transform(df['Sex']) # 규칙은 함수에서 적용됨
p(label_encoder.classes_)
p(df.head())

# 레이블 인코딩용 함수
def encode_features(df, features): # 범용적으로 사용하기 위해 features는 밖에 쓰기
    for i  in features:
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])
    return df # feature들이 레이블 인코딩된 데이터 프레임을 반환

features = ['Sex', 'Cabin', 'Embarked']
p(encode_features(df, features)) # 잘 변환됨을 확인할 수 있다.
~~~

~~~python
## crossvaildation.py

## 교차 검증 모델 학습 및 성능 평가

def p(str):
    print(str, '\n')

# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

# 데이터 프레임 생성
df = pd.read_csv('./assets/train.csv')
#p(df.info())

# 피처 / 레이블 -> 분리
features = df.drop('Survived', axis=1) # 피처
labels = df['Survived'] # 레이블
# p(features)
# p(labels)

# features 데이터 전처리
# 결측치 제거, 불필요한 속성 제거 (Passenger Id, Name, Ticket)

# null 개수 확인
p(features.isnull().sum())

# 결측치 제거 (null 값 대체)
features["Age"].fillna(features["Age"].mean(), inplace=True)
features['Cabin'].fillna('N', inplace=True)
features['Embarked'].fillna('N', inplace=True)

# 불필요한 속성 제거 (PassengerId, Name, Ticket)
features = features.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
#p(features) # 필요 없는 정보는 뺴기

# 레이블 인코딩 진행 -> 숫자화 필수
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['Sex'] = le.fit_transform(features['Sex'])
features['Embarked'] = le.fit_transform(features['Embarked'])
features['Cabin'] = le.fit_transform(features['Cabin'])
# p(features) # 모든 값을 숫자로 변환한 것. 머신러닝을 위해

# 훈련 세트와 테스트 세트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features, # 피처
    labels, # 레이블
    test_size=0.2, # 분할 비율(20%가 테스트 세트, 80%가 훈련 세트라는 의미 )
    random_state=53 # 랜덤 시드 값
)

# Shape 확인 -> 크기 확인
# p(X_train.shape) # (712, 8)
# p(X_test.shape) # (179, 8)
# p(y_train.shape) # (712,)
# p(y_test.shape) # (179,)

## 모델 학습

# 의사결정 나무 -> 이분화하는 방식
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 의사결정트리 분류기
dt = DecisionTreeClassifier(random_state=11)
# p(dt)

# 모델 학습
dt.fit(X_train, y_train)

# 예측치
pred = dt.predict(X_test)

# 정확도 -> 많이 사용하게 될 함수이니, 기억을 해두자.
p(f"정확도: {accuracy_score(y_test, pred)}")

## 교차 검증

# 라이브러리 로딩
from sklearn.model_selection import cross_val_score

# dt: 데이터 프레임, features: 피처, labels: 레이블, cv: 폴드 수
scores = cross_val_score(dt, features, labels, cv=10)

# 교차 검증마다 정확도 구하기
# iter_count: 반복 횟수, accuracy: 정확도
# enumerate(scores): 교차 검증 결과를 열거
for iter_count, accuracy in enumerate(scores):
    p(f"{iter_count + 1}번째 교차 검증 정확도: {accuracy}")

# 평균 정확도
p(np.mean(scores))

# 10번: 0.7811860174781523
# 20번: 0.7756565656565656
# 50번: 0.7949673202614378
# 100번: 0.7861111111111111 
# 이 결과를 확인한 결과, 폴드의 수를 늘린다고 정확도가 올라가지는 않는다.
~~~




