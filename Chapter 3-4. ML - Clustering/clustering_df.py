## clustering_df.py

def p(str):
    print(str, '\n')

# 필요한 라이브러리 로딩
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# rent.csv 데이터 로딩
data = pd.read_csv('./assets/rent.csv')
#data.info()

# rent.csv변수
'''
Posted On 게시된 날짜
BHK 침실, 홀, 주방의 수
Rent 임대료
Size 크기
Floor 층수
Area Type 주택이 속한 지역의 유형
Area Locality 지역의 위치
City 도시의 이름
Furnishing Status 추택 또는 아파트의 가구 유무 상태
Tenant Preferred 선호하는 임차인 유형
Bathroom 욕실수 
Point of Contact 문의할 담당자 
'''

# 범주형 변수 분석
# p(data['Floor'].value_counts()) # 값별로 몇 개가 있는지 확인할 수 있다.
# p(data['Area Type'].value_counts())
# p(data['Area Locality'].value_counts())
# p(data['City'].value_counts())
# p(data['Furnishing Status'].value_counts())
# p(data['Point of Contact'].value_counts())

# 컬럼명 변경 -> 아직 바꾼 것은 아니고, Mapping을 한 것이다.
new_column_name = {
    "Posted On":"Posted_On",
    "BHK":"BHK",
    "Rent":"Rent",
    "Size":"Size",
    "Floor" : "Floor",
    "Area Type" : "Area_Type",
    "Area Locality" : "Area_Locality",
    "City":"City",
    "Furnishing Status":"Furnishing_Status",
    "Tenant Preferred":"Tenant_Preferred",
    "Bathroom":"Bathroom",
    "Point of Contact":"Point_of_Contact"
}
data.rename(columns=new_column_name, inplace = True)

# BHK 값들을 오름차순으로 정렬
data['BHK'].sort_values()

# Rent 확인
p(data['Rent'].value_counts())
p(data['Rent'].sort_values())
# 아웃라이어
# 데이터 집합에서 다른 관측치들과 동떨어진 극단적인 값을 가지는 데이터 포인트
# -> 외톨이라 생각하자.

# Rent boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=data['Rent'])
# plt.show()
# 대부분의 데이터는 범위안에 존재하나, 하나의 데이터가 3.5 부분에 있다. 즉 매우 동떨어진 것을 확인할 수 있다.

# Rent Scatter
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=data.index, y=data['Rent'])
# plt.show()
# 마찬가지로 이상치가 존재함을 확인할 수 있다.
# 아웃라이어를 확인할 때는 Boxplot이나 Scatter를 확인하면 된다.

# BHK와 Rent의 상관관계
corr_Br = data['BHK'].corr(data['Rent'])
# p(f"BHK와 Rent의 상관관계: {corr_Br:.2f}")
# 상관관계가 0.37이다.

# 산점도 시각화
# plt.scatter(data['BHK'], data['Rent'])
# plt.grid(True)
# plt.show()

# size 확인
# p(data['Size'].value_counts())
# p(data['Size'].sort_values())

#size displot
# sns.displot(data['Size'])
# plt.show()

# Size와 Rent의 관계
# plt.scatter(data['Size'], data['Rent'])
# plt.show()
# 데이터를 직접보는 것은 사실 쉽지 않기에 그래프로 확인하는 것이다.

# 상관관계 확인
# 1. 임대료와 BHK 상관관계
p(f"BHK와 임대료 상관관계 : {data['BHK'].corr(data['Rent'])}")

# 2. 임대료와 Size 상관관계
p(f"Size와 임대료 상관관계 : {data['Size'].corr(data['Rent'])}")

# 3. 임대료와 도시별 상관관계
# 도시는 문자이므로 수치형으로 변환시켜야 한다.
cites = data['City'].unique() # 유일한 도시값들
# p(cites) -> 유일하게 하나씩 그룹핑함
for city in cites:
    city_data = data[data['City'] == city]
# city로 그룹핑한 후 각 그룹들의 평균 임대료
city_mean = data.groupby('City')['Rent'].mean()
# p(city_mean)

data['City_Mean'] = data.groupby('City')['Rent'].transform('mean')
# p(data['City_Mean'])

# 상관관계 확인
# p(f"임대료와 도시별 평균 임대료의 상관관계 : {data['Rent'].corr(data['City_Mean'])}")

# City, Rent 그룹과 Rent 상관관계
rent_city = data.groupby('City')['Rent'].corr(data['Rent'])
# p(rent_city)

# 도시목록 가져오기
cites = data['City'].unique()
# p(cites)

# 도시별 임대료 상관관계
city_rent_corr = {}
for i in cites:
    city_data = data[data['City'] == i]
    correlation = city_data['Rent'].corr(city_data['Rent'])
    city_rent_corr[i] = correlation

# p(city_rent_corr)

# 수치형 데이터들로 heatmap 그리기

#수치형 변수만 선택
# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# numeric_data.corr()
# plt.figure(figsize = (10,8))
# sns.heatmap(numeric_data.corr(), annot=True)
# plt.show()

# 지역별 임대료
# plt.figure(figsize = (10,6))
# sns.boxplot(x='City', y='Rent', data=data)
# plt.grid(True)
# plt.show()

# 평균 임대료 계산
avg_rent_city = data.groupby('City')['Rent'].mean().sort_values(ascending=False)
p(avg_rent_city)

# 날짜 데이터 변환
data['Posted_On'] = pd.to_datetime(data['Posted_On'])
data['Year'] = data['Posted_On'].dt.year
data['Month'] = data['Posted_On'].dt.month
# data['Day'] = data['Posted_On'].dt.day

# p(data['Year'].value_counts())
# p(data['Month'].value_counts())

# 월별 평균 임대료
avg_month_rent = data.groupby(['Year', 'Month'])['Rent'].mean()
# p(avg_month_rent)

# 월별 평균 임대료 시각화
# plt.figure(figsize=(12, 6))
# avg_month_rent.plot(kind='line', marker='o')
# plt.grid(True)
# plt.show()

# 모델 선택
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 필요한 컬럼 선택
features = ['BHK', 'Size', 'Floor', 'Bathroom']
data1 = data[features + ['Rent']]
p(data1)

# Floor 컬럼의 데이터 전처리 : 문자열에서 숫자만 추출하여 float으로 변환
data1['Floor'] = data1['Floor'].str.extract('(\d+)').astype(float)
# p(data1['Floor'])

# 결측치 처리
data1 = data1.dropna() # 결측치가 있는 행 삭제
data1.info()

# 훈련세트와 테스트세트 분리
X = data1[features]
y = data1['Rent']
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형회귀 모델 생성
lr = LinearRegression()

# 훈련
lr.fit(X_train, y_train)

# 예측값
pred = lr.predict(X_test)

# 평균 제곱 오차 계산 (MSE)
mse = mean_squared_error(y_test, pred)
p(f"평균 제곱 오차: {mse}") #2542273917.555011

# 실제 값과 예측 값 시각화
plt.figure(figsize=(12, 6))
plt.scatter(y_test, pred)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--'
)
plt.grid(True)
plt.show()