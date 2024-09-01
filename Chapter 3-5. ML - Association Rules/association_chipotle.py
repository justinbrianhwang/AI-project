# association_chipotle.py

## 연관규칙학습
def p(str):
    print(str, '\n')

# 라이브러리 로딩
import pandas as pd
from mlxtend import frequent_patterns
from mlxtend import preprocessing

# 데이터 프레임
df = pd.read_csv('./assets/chipotle.csv')
df.info()
p(df)

# chipotle.csv
'''
order_id : 주문 번호 
quantity : 수량
item_name : 아이템명 
choice_description : 선택옵션 
item_price : 아이템 가격 
'''

# 연관분석 함수를 사용하기 위해 리스트로 변환
df_temp = df[['order_id', 'item_name']]
p(df_temp)
df_temp_arr = [[] for i in range(1835)]
p(df_temp_arr)
num = 0
for i in df_temp['item_name']:
    df_temp_arr[df_temp['order_id'][num]].append(i)
    num += 1
p(df_temp_arr)

# order_id는 1부터 시작하므로 빈값 제거
df_temp_arr.pop(0)
p(df_temp_arr) # 빈리스트 제거 완료

# set을 사용하여 중복값 제거 후 list로 반환
num = 0
for i in df_temp_arr:
    df_temp_arr[num] = list(set(df_temp_arr[num]))
    num += 1
p(df_temp_arr) # 중복값을 날렸다.

# unique한 아이템명
p(df['item_name'].unique())

# 트랜잭션 인코더 생성
te = preprocessing.TransactionEncoder()

# 학습시키고, 변환
te_arr = te.fit(df_temp_arr).transform(df_temp_arr)
p(te_arr)

# 변환된 이진행렬을 데이터 프레임으로 변환 후
# 각 열의 이름을 아이템명으로 설정
df1 = pd.DataFrame(te_arr, columns=te.columns_)
p(df1)

# 정규표현식 라이브러리
import re

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

# 가격에 '$' 기호가 있는데, 이를 없앨 것이다. 1) 정규표현식을 사용하자.
num = 0
for i in df['item_price']:
    df['item_price'][num] = re.sub(pattern='[$]', repl='', string=i)
    num += 1
p(df) # 데이터 전처리를 완료했다. 우리가 원하는 대로 알고리즘을 사용하려면 전처리는 필수이다.

# 가격에 '$' 기호가 있는데, 이를 없앨 것이다. 2) lambda함수 사용
# apply() : 앞에있는 것 하나하나에 뒤에 람다함수를 적용
df['item_price'] = df['item_price'].apply(lambda x:x.lstrip('$')) # 람다함수를 사용하면 한줄로 끝낼 수 있다.
p(df)

# 아이템 가격을 float로 변환
df['item_price'] = df['item_price'].astype(float)
p(df)
p(df['item_price'].sum()) # 숫자 변환 확인 방법은 다 더해보면 된다.

# null인 것의 합계
p(df.isnull().sum())

# null인 것들을 default로 채움
df['choice_description'] = df['choice_description'].fillna('default')
p(df.isnull().sum())

# 알파벳과 쉼표를 제외한 모든 문자 제거
num = 0
for i in df['choice_description']:
    df['choice_description'][num] = \
        re.sub(pattern='[^a-zA-Z,]', repl='', string=i)
    num += 1
p(df['choice_description'])

# 아이템명과 선택옵션의 값의 수를 시리즈로
result = df.groupby(['item_name', 'choice_description']).value_counts()
p(result)

# 시리즈의 인덱스로 리스트 변환
temp_index = result.index.tolist()

# 시리즈의 값들을 리스트 변환
temp_values = result.values.tolist()
p(temp_values)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
x = df['item_name']
y = df['quantity']
plt.bar(x, y)
plt.xticks(rotation=45)
plt.show()

# 중복된 아이템 제거하고 판매량 합계 계산
p(df.groupby('item_name')['quantity'].sum())

# 가장 많이 팔린 메뉴 중 10개 막대그래프
top_items \
    = df.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(10)
p(top_items)

# nlargest 함수 (가장 큰 값을 가지는 아이템 추출) 사용해서 상위 10개 아이템 추출
top_items = df.groupby('item_name')['quantity'].sum().nlargest(10)
p(top_items)
# 위의 두 방법은 같은 것이다.

plt.figure(figsize=(10, 6))
top_items.plot(kind='barh')
plt.xlabel('item_name')
plt.ylabel('quantity')
plt.show()

# 연관학습 라이브러리
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# 그룹핑 후 확인
# unstack : 인덱스 값을 컬럼으로 사용
df_grouped = df.groupby(['order_id', 'item_name'])['quantity']\
    .sum().unstack().reset_index()
p(df_grouped)

# 판매량이 1이상인 경우 1로 변경
df_group = df_grouped.apply(lambda x: 1 if x >= 1 else 0)
# p(df_group)

# apriori 알고리즘을 사용해 연관규칙 학습
frequent = apriori(
    df_group.drop('order_id', axis = 1),
    min_support= 0.01, #최소 지지율
    use_colnames= True # 컬럼명 사용여부
)

# 연관규칙 추출
#lift : 향상도 (규칙이 얼마나 의미있는지)
rules = association_rules(
    frequent,
    metric='lift',
    min_threshold = 1.0
)
p(rules)

# 연관학습 결과
p(rules['support'])
