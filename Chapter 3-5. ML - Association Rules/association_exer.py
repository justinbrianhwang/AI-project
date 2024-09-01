## association_exer.py

## 연관규칙 실습

# association.py, association_chipotle.py 파일을 참조하여
# store_data.csv 또는 groceries.csv 파일을 활용해
# apriori 알고리즘을 활용해 연관규칙을 도출해 내고 support, confidence 확인

# 라이브러리 import
import mlxtend
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def p(str):
    print(str, '\n')
# -------------------------------------------------------------------------------------------------------
# 데이터 프레임
store_df = pd.read_csv('./assets/store_data.csv', header=None)

# 결측치를 문자열로 변환 (예: NaN -> 'None' 또는 빈 문자열)
store_df = store_df.fillna('')

# 모든 데이터를 문자열로 변환
store_df = store_df.astype(str)

store_na = store_df.to_numpy()
p(store_na)

# 트랜잭션 인코더
te = TransactionEncoder()

# 데이터 학습 후 변환
te_array = te.fit(store_na).transform(store_na)
p(te_array)

# 데이터 프레임 만들기
df = pd.DataFrame(te_array, columns=te.columns_)
p(df)

# 지지도, 신뢰도 설정
min_support_per = 0.2
min_threshold = 0.5

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

# p(result_rules)
p(result_rules['support'])
p(result_rules['confidence'])


















































































