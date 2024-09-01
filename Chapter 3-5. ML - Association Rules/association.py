## association.py
## 연관규칙학습

def p(str):
    print(str, '\n')

# mlxtend 라이브러리 설치
import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# itemset 불러오기
# 리스트를 numpy 배열로 바꾸기
data = np.array([
    ['우유', '기저귀', '쥬스', ''],
    ['상추', '기저귀', '맥주', ''],
    ['우유', '양상추', '기저귀', '맥주'],
    ['양상추', '맥주', '', '']
])

# 트랜잭션 인코더
te = TransactionEncoder()

# 데이터 학습 후 변환
te_array = te.fit(data).transform(data)

# 데이터 프레임 만들기
df = pd.DataFrame(te_array, columns=te.columns_)
p(df)

# 지지도, 신뢰도 설정
min_support_per = 0.5 # 최소 지지도 설정 (기본 0.5)
min_threshold = 0.5 # 최소 신뢰도 설정 (기본 0.8)

# apriori 알고리즘 적용
result = apriori(
    df, # 데이터 프레임
    min_support=min_support_per, # 최소 지지도
    use_colnames=True # 컬럼명 사용 여부
)

# 연관 규칙 생성
result_rules = association_rules(
    result, # 알고리즘 적용 결과
    metric='confidence', # 향상도
    min_threshold=min_threshold # 최소 신뢰도
)

p(result_rules)
p(result)
p(result_rules['support']) # 지지도
p(result_rules['confidence']) # 신뢰도 -> 신뢰도를 구하는 것이 핵심이다.

