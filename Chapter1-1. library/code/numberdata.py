### numberdata.py

## 숫자데이터 처리 라이브러리

def p(str):
    print(str, "\n")

# math
import math
p(math.gcd(60, 80, 100)) # 최대공약수
p(math.lcm(15, 25)) # 최소공배수

# decimal
from decimal import Decimal
p(0.1 * 3) # 메모리의 한계로 인해서 소수점 연산 부정확함
p(Decimal('0.1') * 3) # 정확한 소수연산시에 Decimal 사용

# fractions
from fractions import Fraction
p(Fraction(1.5)) # 분수

# random
import random
p(random.randint(1, 45)) # 1~45까지의 랜덤 정수

# 로또 숫자 6개 추출
lottoNum = set()
while True:
    lottoNum.add(random.randint(1, 45))
    if (len(lottoNum)==6):
        break
p(list(lottoNum))

# statistics
import statistics
score = [38, 54, 45, 87, 92]
p(statistics.mean(score)) # 평균값
p(statistics.median(score)) # 중간값 (=중앙값)




















