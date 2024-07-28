# English

There are two types of libraries:

1. Built-in libraries
2. Installed (external) libraries

We will briefly cover these two types.

# Python 3 Language Reference

http://docs.python.org/3/reference

# Python 3 Standard Library

https://docs.python.org/3/library

→ Referencing the following information will be helpful.

### Text Data

Text data is used very frequently in data processing, so there are many libraries available.

- **textwrap**: Used for formatting text
- **re**: Used for handling text with regular expressions

*Regular expressions are a powerful tool supported by all programming languages. Using them well can make your code more concise.

[stringdata.py] → Including explanation

```python
# Text processing library

# Built-in libraries are provided by Python and do not require separate installation.
# External libraries need to be installed separately after installing Python.
# Use the import statement to use a library.

# Print function
def p(str):
    print(str, "\n")

## textwrap
import textwrap

str = "Hello Python"
# Shorten the string (string, width, placeholder)
p(textwrap.shorten(str, width=10, placeholder="..."))

str = str * 10  # Repeat string 10 times
p(str)
# Convert the string to a list of 11 elements based on whitespace
wrapstr = textwrap.wrap(str, width=11)
p(wrapstr)

# Convert each element of the list to a string with a newline character
p("\n".join(wrapstr))

## re (regular expression)
# Used to search, extract, and replace substrings in a string using a combination
# of pattern strings and flag strings
# Regular expressions are commonly used across all programming languages, so it is essential to learn them!

import re
str = "Hong Gil-dong's phone number is 010-1234-5678"
pattern = re.compile(r"(\d{3})-(\d{4})-(\d{4})")
p(pattern.sub(r"\g<1> \g<2> \g<3>", str))

# Create patterns for phone numbers, emails, IP addresses, social security numbers, etc.

# Regular expression examples

# 1. Search for "apple" in the string
text = "I like apple pie"
result = re.findall(r"apple", text)
p(result)

# 2. Search for one or more digits
text = "My phone number is 010-1234-5678."
result = re.findall(r"\d+", text)
p(result)

# 3. Simple email address pattern search
text = "email address is example@email.com"
result = re.findall(r"\b\w+@\w+\.\w+", text)
p(result)

# 4. Simple phone number pattern search
text = "Call me at 010-1234-5678"
result = re.findall(r"\d{3}-\d{4}-\d{4}", text)
p(result)

# 5. Search for uppercase English letters
text = "Hello Python"
result = re.findall(r"[A-Z]", text)
p(result)

# 6. Remove unnecessary whitespace in a string
text = "Hello     Python     This is Me"
result = re.sub(r"\s+", " ", text)
p(result)

# 7. Search for the start and end of a string
# ^: start, [^]: negation  ex) ^a: starts with 'a', [^a]: not 'a'
text = "Hello World"
result = re.findall(r"^Hello|World$", text)
p(result)

# 8. Search for strings that start with a specific word
# \b : word boundary
text = "Start your journey with a smile. Start early to avoid traffic."
result = re.findall(r"\bStart\b[^.]*\.", text)
p(result)

# 9. Search for URLs in a string
text = "Website URL is http://example.com or https://www.example.com"
result = re.findall(r"https?://[^\s]+", text)
p(result)

# 10. Search for date formats (YYYY-MM-DD)
text = "Today is 2024-07-27 and tomorrow is 2024-07-28"
result = re.findall(r"\d{4}-\d{2}-\d{2}", text)
p(result)
```

### Numeric Data

Libraries used for handling numeric data:

- **math**: Mathematics-related functions
- **decimal**: Handling decimal numbers
- **fractions**: Handling fractions
- **random**: Generating random numbers
- **statistics**: Calculating averages, medians, etc.

[numberdata.py] → Including explanation

```python
# numberdata.py

# Numeric data processing libraries

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
```

### Date Data

Libraries related to handling date data:

- **datetime**
- **calendar**

[datedata.py] → Including explanation

```python
# datedata.py

# Date data processing

def p(str):
    print(str, "\n")

# datetime
import datetime
today = datetime.date.today()
p(today)
p(today.weekday())  # Day of the week
p(today + datetime.timedelta(days=100))  # 100 days later
p(today + datetime.timedelta(days=-100))  # 100 days before
p(today + datetime.timedelta(weeks=3))  # 3 weeks later
p(today + datetime.timedelta(hours=45))  # 45 hours later

day1 = datetime.date(2019, 1, 1)
day2 = datetime.date(2024, 7, 27)
p(day2 - day1)  # Date interval

# calendar
import calendar
p(calendar.weekday(2024, 7, 27))  # Day of the week
p(calendar.isleap(2024))  # Leap year check
```

### Object Data

There are various libraries for handling object data:

- **pickle**
- **shelve**

[objectdata.py] → Including explanation

```python
# objectdata.py

# Object data handling

## pickle
import pickle
obj = {
    "name": "Hong Gil-dong",
    "age": 20
}
# Write the data of obj object to obj.obj file as binary
# wb : binary write mode
with open('obj.obj', 'wb') as f:
    pickle.dump(obj, f)
# Read binary data from obj.obj file
# rb : binary read mode
with open('obj.obj', 'rb') as f:
    print(pickle.load(f))

## shelve
import shelve
def save(key, value):
    with shelve.open("shelve") as f:
        f[key] = value
def get(key):
    with shelve.open("shelve") as f:
        return f[key]

save("number", [1, 2, 3, 4, 5])
save("string", ["a", "b", "c"])
print(get("number"))
print(get("string"))
```

### Format Data

Various libraries support handling formatted data:

- **csv**
- **xml**
- **json**

[formatdata.py] → Including explanation

```python
# formatdata.py

# Format data: The format of data exchanged over a network
# Data: text (CSV, XML, JSON ...), binary (jpg, mp4 ...)
# Network data
# 1. CSV (Comma Separated Value)
# 2. XML (Extensible Markup Language)
#    Advantages: Data structure + data, Disadvantages: Uses many bytes to represent data → High network cost
# 3. JSON (JavaScript Object Notation)
#    Advantages: Reduces network cost, Disadvantages: Cannot represent data structures
# 4. XML + JSON

# CSV
import csv
with open('csvdata.csv', mode='w', encoding='utf-8') as f:
    # delimiter: data separator
    # quotechar: character recognized as string
    writer = csv.writer(f, delimiter=',', quotechar="'")
    # Write row to csv file
    writer.writerow(['Hong Gil-dong', '30', 'Seoul'])
    writer.writerow(['Gang Gam-chan', '40', 'Busan'])
with open('csvdata.csv', mode='r', encoding='utf-8') as f:
    print(f.read())

# xml
import xml.etree.ElementTree as ET

persons = ET.Element("persons")
person = ET.SubElement(persons, "person")
name = ET.SubElement(person, 'name')
name.text = "Hong Gil-dong"
age = ET.SubElement(person, 'age')
age.text = "20"
person = ET.SubElement(persons, "person")
name = ET.SubElement(person, 'name')
name.text = "Gang Gam-chan"
age = ET.SubElement(person, 'age')
age.text = "30"

xmlstr = ET.tostring(persons, encoding="utf-8").decode()
print(xmlstr)

with open("xmldata.xml", mode="w", encoding="utf-8") as f:
    writer = f.write(xmlstr)

with open("xmldata.xml", mode="r", encoding="utf-8") as f:
    print(f.read())
```

### Remote Json Data

- **requests**
- **urllib**
- **aiohttp**

[remotejsondata.py] → Including explanation

```python
# remotejsondata.py

from util import p

# requests
import requests
import json

response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = response.json()
p(data)

# urllib
import json
from urllib.request import urlopen

response = urlopen('https://jsonplaceholder.typicode.com/posts')
if response.getcode() == 200:
    data = json.loads(response.read().decode('utf-8'))
    for post in data:
        p(post['title'])
else:
    p('Error occurred!')

# aiohttp
import aiohttp
import asyncio
import json

async def fetch_json(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return data

async def main():
    url = 'https://jsonplaceholder.typicode.com/posts'
    data = await fetch_json(url)
    p(json.dumps(data, indent=4))

asyncio.run(main())
```

# Korean 

라이브러리의 종류에는 두 가지가 있다.

1. 내장 라이브러리
2. 설치(외부) 라이브러리

이 두 가지에 대해 간단히 다뤄볼 것이다.

# 파이썬3 언어 래퍼런스

http://docs.python.org/3/reference

# 파이썬3 표준 라이브러리

https://docs.python.org/3/library

→ 다음 정보를 반드시 참고하면서 작업을 하면 도움이 될 것이다.

### 문자 데이터

문자 데이터는 데이터 처리 시 가장 많은 빈도로 사용되므로 매우 다양한 라이브러리가 존재함

- **textwrap**: 문자열 가공에 사용됨
- **re**: 정규 표현식을 이용한 문자열 처리

*정규 표현식의 경우는 모든 언어에서 지원하는 강력한 도구이다. 이를 잘 사용하면 코드를 간결히 쓸 수 있다.

[stringdata.py] → 설명까지 추가

```python
# 문자열 처리 라이브러리

# 내장 라이브러리는 파이썬에서 기본 제공하는 라이브러리로, 별도 설치가 불필요함
# 외부 라이브러리는 파이썬 설치 후에 별도로 설치해야 사용 가능
# 라이브러리 사용 시에는 import 구문 사용

# 출력 함수
def p(str):
    print(str, "\n")

## textwrap
import textwrap

str = "Hello Python"
# 문자열 길이 축약 (문자열, 길이, 대체 문자열)
p(textwrap.shorten(str, width=10, placeholder="..."))

str = str * 10  # 문자열 10번 반복
p(str)
# 문자열 공백 기준으로 11개 요소로 리스트 변환
wrapstr = textwrap.wrap(str, width=11)
p(wrapstr)

# 리스트의 각 요소에 줄바꿈 문자를 붙여서 문자열로 변환
p("\n".join(wrapstr))

## re (정규 표현식)
# 문자열 전체에서 부분 문자열들을 탐색, 추출, 대체하는 데 사용되는
# 패턴 문자열과 플래그 문자열의 조합인 식
# 모든 프로그래밍 언어에서 공통적으로 사용되는 식이므로 학습 필수!

import re
str = "홍길동의 전화번호는 010-1234-5678"
pattern = re.compile(r"(\d{3})-(\d{4})-(\d{4})")
p(pattern.sub(r"\g<1> \g<2> \g<3>", str))

# 전화번호, 이메일, IP 주소, 주민등록번호 등 자주 사용되는 패턴을 만들어 봅시다!

# 정규 표현식 예제

# 1. 문자열에서 apple을 검색
text = "I like apple pie"
result = re.findall(r"apple", text)
p(result)

# 2. 1개 이상의 숫자를 검색
text = "My phone number is 010-1234-5678."
result = re.findall(r"\d+", text)
p(result)

# 3. 간단한 이메일 주소 패턴 검색
text = "email address is example@email.com"
result = re.findall(r"\b\w+@\w+\.\w+", text)
p(result)

# 4. 간단한 휴대폰 번호 패턴 검색
text = "Call me at 010-1234-5678"
result = re.findall(r"\d{3}-\d{4}-\d{4}", text)
p(result)

# 5. 영문 대문자 패턴 검색
text = "Hello Python"
result = re.findall(r"[A-Z]", text)
p(result)

# 6. 문자열 내의 불필요한 공백 제거
text = "Hello     Python     This is Me"
result = re.sub(r"\s+", " ", text)
p(result)

# 7. 문자열의 시작과 끝 검색
# ^: 시작, [^]: 부정  ex) ^a: a문자로 시작, [^a]: a문자가 아님
text = "Hello World"
result = re.findall(r"^Hello|World$", text)
p(result)

# 8. 특정 단어로 시작하는 문자열 검색
# \b : word(\w, 알파벳 대소문자 또는 숫자 또는 _)의 경계
text = "Start your journey with a smile. Start early to avoid traffic."
result = re.findall(r"\bStart\b[^.]*\.", text)
p(result)

# 9. 문자열에서 URL 검색
text = "Website URL is http://example.com or https://www.example.com"
result = re.findall(r"https?://[^\s]+", text)
p(result)

# 10. 날짜 형식 (년도 네 자리-월 두 자리-일 두 자리) 검색
text = "오늘은 2024-07-27일 이고 내일은 2024-07-28일"
result = re.findall(r"\d{4}-\d{2}-\d{2}", text)
p(result)
```

### 숫자 데이터

숫자 데이터를 처리하는 데 사용되는 라이브러리:

- **math**: 수학 관련 함수
- **decimal**: 소수점 처리
- **fractions**: 분수 처리
- **random**: 랜덤한 수 추출
- **statistics**: 평균값, 중간값 계산

[numberdata.py] → 설명까지 추가

```python
# numberdata.py

# 숫자 데이터 처리 라이브러리

def p(str):
    print(str, "\n")

# math
import math
p(math.gcd(60, 80, 100))  # 최대 공약수
p(math.lcm(15, 25))  # 최소 공배수

# decimal
from decimal import Decimal
p(0.1 * 3)  # 메모리의 한계로 인해서 소수점 연산 부정확함
p(Decimal('0.1') * 3)  # 정확한 소수 연산 시에 Decimal 사용

# fractions
from fractions import Fraction
p(Fraction(1.5))  # 분수

# random
import random
p(random.randint(1, 45))  # 1~45까지의 랜덤 정수

# 로또 숫자 6개 추출
lottoNum = set()
while True:
    lottoNum.add(random.randint(1, 45))
    if len(lottoNum) == 6:
        break
p(list(lottoNum))

# statistics
import statistics
score = [38, 54, 45, 87, 92]
p(statistics.mean(score))  # 평균값
p(statistics.median(score))  # 중간값
```

### 날짜 데이터

날짜 데이터를 처리하는 데 사용되는 라이브러리:

- **datetime**
- **calendar**

[datedata.py] → 설명까지 추가

```python
# datedata.py

# 날짜 데이터 처리

def p(str):
    print(str, "\n")

# datetime
import datetime
today = datetime.date.today()
p(today)
p(today.weekday())  # 요일
p(today + datetime.timedelta(days=100))  # 100일 후
p(today + datetime.timedelta(days=-100))  # 100일 전
p(today + datetime.timedelta(weeks=3))  # 3주 후
p(today + datetime.timedelta(hours=45))  # 45시간 후

day1 = datetime.date(2019, 1, 1)
day2 = datetime.date(2024, 7, 27)
p(day2 - day1)  # 날짜 간격

# calendar
import calendar
p(calendar.weekday(2024, 7, 27))  # 요일
p(calendar.isleap(2024))  # 윤년 여부
```

### 객체 데이터

객체 데이터를 다루는 다양한 라이브러리:

- **pickle**
- **shelve**

[objectdata.py] → 설명까지 추가

```python
# objectdata.py

# 객체 데이터 처리

## pickle
import pickle
obj = {
    "name": "홍길동",
    "age": 20
}
# obj 객체의 데이터를 obj.obj 파일에 바이너리로 쓰기
# wb: 바이너리 쓰기 모드
with open('obj.obj', 'wb') as f:
    pickle.dump(obj, f)
# obj.obj 파일에서 바이너리 데이터를 읽기
# rb: 바이너리 읽기 모드
with open('obj.obj', 'rb') as f:
    print(pickle.load(f))

## shelve
import shelve
def save(key, value):
    with shelve.open("shelve") as f:
        f[key] = value
def get(key):
    with shelve.open("shelve") as f:
        return f[key]

save("number", [1, 2, 3, 4, 5])
save("string", ["a", "b", "c"])
print(get("number"))
print(get("string"))
```

### 포맷 데이터

포맷팅된 데이터를 처리하는 다양한 라이브러리 지원:

- **csv**
- **xml**
- **json**

[formatdata.py] → 설명까지 추가

```python
# formatdata.py

# 포맷 데이터: 네트워크상에서 주고 받는 데이터의 형식
# 데이터: 문자(CSV, XML, JSON ...), 바이너리(jpg, mp4 ...)
# 네트워크상의 데이터
# 1. CSV (Comma Separated Value, 콤마로 구분된 값)
# 2. XML (Extensible Markup Language, 확장 가능한 표기 언어)
#    장점: 데이터 구조 + 데이터, 단점: 데이터 표현에 많은 바이트 사용 → 네트워크 비용 크다
# 3. JSON (JavaScript Object Notation, 자바스크립트 객체 표기법)
#    장점: 네트워크 비용 절감, 단점: 데이터 구조 표현 불가
# 4. XML + JSON

# CSV
import csv
with open('csvdata.csv', mode='w', encoding='utf-8') as f:
    # delimiter: 데이터 구분자
    # quotechar: 문자열로 인식하는 문자
    writer = csv.writer(f, delimiter=',', quotechar="'")
    # csv 파일에 행 쓰기
    writer.writerow(['홍길동', '30', '서울'])
    writer.writerow(['강감찬', '40', '부산'])
with open('csvdata.csv', mode='r', encoding='utf-8') as f:
    print(f.read())

# xml
import xml.etree.ElementTree as ET

persons = ET.Element("persons")
person = ET.SubElement(persons, "person")
name = ET.SubElement(person, 'name')
name.text = "홍길동"
age = ET.SubElement(person, 'age')
age.text = "20"
person = ET.SubElement(persons, "person")
name = ET.SubElement(person, 'name')
name.text = "강감찬"
age = ET.SubElement(person, 'age')
age.text = "30"

xmlstr = ET.tostring(persons, encoding="utf-8").decode()
print(xmlstr)

with open("xmldata.xml", mode="w", encoding="utf-8") as f:
    writer = f.write(xmlstr)

with open("xmldata.xml", mode="r", encoding="utf-8") as f:
    print(f.read())
```

### 원격 Json 데이터

- **requests**
- **urllib**
- **aiohttp**

[remotejsondata.py] → 설명까지 추가

```python
# remotejsondata.py

from util import p

# requests
import requests
import json

response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = response.json()
p(data)

# urllib
import json
from urllib.request import urlopen

response = urlopen('https://jsonplaceholder.typicode.com/posts')
if response.getcode() == 200:
    data = json.loads(response.read().decode('utf-8'))
    for post in data:
        p(post['title'])
else:
    p('에러 발생!')

# aiohttp
import aiohttp
import asyncio
import json

async def fetch_json(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return data

async def main():
    url = 'https://jsonplaceholder.typicode.com/posts'
    data = await fetch_json(url)
    p(json.dumps(data, indent=4))

asyncio.run(main())
```




