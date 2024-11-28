### numberdata.py

## Numeric Data Processing Libraries

def p(str):
    print(str, "\n")

# math
import math
p(math.gcd(60, 80, 100))  # Greatest common divisor
p(math.lcm(15, 25))  # Least common multiple

# decimal
from decimal import Decimal
p(0.1 * 3)  # Inaccurate decimal arithmetic due to memory limitations
p(Decimal('0.1') * 3)  # Accurate decimal arithmetic using Decimal

# fractions
from fractions import Fraction
p(Fraction(1.5))  # Fraction

# random
import random
p(random.randint(1, 45))  # Random integer between 1 and 45

# Extracting 6 random Lotto numbers
lottoNum = set()
while True:
    lottoNum.add(random.randint(1, 45))
    if len(lottoNum) == 6:
        break
p(list(lottoNum))

# statistics
import statistics
score = [38, 54, 45, 87, 92]
p(statistics.mean(score))  # Mean
p(statistics.median(score))  # Median
