# English

## Data Types

Data types are a critical component in programming, as they determine the amount of memory allocated for data storage.

### Python Data Types
1. **Number**: Integers, floats, octal, hexadecimal
   - Unlike other languages, Python does not distinguish between integers and floats. They are all considered numbers.

2. **String**: Characters and strings (arrays of characters)
   - In Python, there is no distinction between a character and a string.

3. **List**: Stores literals of various types, mutable, allows duplicates
   - Uses ‘[’ and ‘]’ symbols.

4. **Tuple**: Stores literals of various types, immutable, allows duplicates
   - Uses ‘(’ and ‘)’ symbols.

5. **Dictionary**: Stores data in the form of Key:Value pairs, keys must be unique, values can be duplicated.

6. **Set**: Stores literals of various types, mutable, does not allow duplicates.

7. **Boolean**: Stores True or False values.

8. **Class**: Data + storage.

[dataTypes.py](./dataTypes.py) code example:
```python
# Print function
def p(value):
    print(value, '\n')

# Number
p(1)
p(0.1)
p(3E5)
p(0o10)
p(0xFF)
p(1+2)
p(3**3)
p(7%4)
p(7//4)

# String
p('Hello')
p("Hello")
p('''
Hello
There
''')
p("Hello 'tom'")
p('Hello "tom"')
p("Hello \"tom\"")
p("Hello\ntom")
p("Hello" + "tom")
p("Hello"*3)
p(len("Hello"))

str = "Hello There"
p(str[0])
p(str[:3])
p(str[3:])
p(str[3:5])
p(str[::2])

str = "jane"
num = 20
p('%s은 %d살 입니다' % (str, num))
p('{0}은 {1}살 입니다'.format(str, num))
p(f'{str}은 {num}살 입니다')
p("%10s" % "hello")
p("%10.4f" % 3.141592)

str = "hello there"
p(str.count('e'))
p(str.find('e'))
p(str.find('k'))
p(str.index('e'))
# p(str.index('k'))
p(str.upper().lower())
p(str.replace("e", "k"))
p(str.split())
p(str.split("e"))

# List
even = [2, 4, 6, 8]
odd = [1, 3, 5, 7]

p(even)
p(odd)

p(even[0])
p(even[0:])
p(even[:2])
p(even[0::2])
p(even[-1])

p(even+odd)
p(even*3)

p(even.index(6))

p(len(even))

del even[0]
p(even)

even.append(10)
p(even)

even.sort()
p(even)

even.reverse()
p(even)

even.insert(0, 2)
p(even)

even.remove(2)
p(even)

even.pop()
p(even)

even.extend([4, 2])
p(even)

# Tuple
# Tuple is like a List but immutable
# Lists use [], Tuples use ()

# Dictionary
# Stores data in Key:Value pairs
# Keys must be unique, Values can be duplicated
# Uses {key:value} syntax

dic = {"name": "홍길동", "age": 20, "address": "서울"}
p(dic)
p(dic["name"])
p(dic.get("name"))

dic["gender"] = "남"
p(dic)

del dic["address"]
p(dic)

p(dic.keys())
p(dic.values())
p(list(dic.keys()))
p(list(dic.values()))

p(dic.items())

dic.clear()
p(dic)

p('address' in dic)

# Set
# Set is a data structure that does not allow duplicate values
# Uses {} syntax

s = {1, 2, 3, 4, 5, 1, 2, 3}
p(s)

s = set(['a', 'b', 'c'])
p(s)

s = set(['hello'])
p(s)

s1 = {1, 2, 3, 4, 5}
s2 = {3, 4, 5, 6, 7}
p(s1 & s2)
p(s1 | s2)
p(s1 - s2)
p(s2 - s1)

# Boolean
# Stores True or False values
# Strings with one or more characters are True
# Lists, Tuples with one or more elements are True
# Dictionaries with one or more items are True
# 0 is False, non-zero is True
# None is False

print(1==1)
print(2<1)
print(bool("hello"))
print(bool(""))
```

## Logic
All algorithms are composed of combinations of "branching" and "looping".

Branching
- if statement

Looping
- for loop

- for in loop

- while loop

``` python

# p function
def p(value):
    print(f"{value}\n")

# if

a = 10
if a > 5:
    p("a는 5보다 큼")
else:
    p("a는 5보다 크지 않음")

a = 10
if a < 5:
    p("a는 5보다 작음")
elif a < 10:
    p("a는 5보다 작지 않지만 10보다 작음")
else:
    p("a는 10보다 작지 않음")

# for

for i in range(1, 10, 2):
    p(i)

l = [1, 2, 3, 4, 5]
for i in l:
    p(i)

l = {'a': 1, 'b': 2, 'c': 3}
for i in l.values():
    p(i)

l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in l:
    if i % 2 == 0:
        continue
    if i == 9:
        break
    else:
        p(i)

# while
# Loops while the condition is True
# There must be a condition that makes the loop False to prevent infinite loops

a = 0
while a < 10:
    p(a)
    a = a + 1

a = 0
while a < 10:
    if a == 5:
        break
    p(a)
    a = a + 1

```


# Korean

## Data Types

데이터 타입은 데이터 저장을 위해 할당하는 메모리의 크기를 결정하는 프로그래밍에서 매우 중요한 구성요소입니다.

### 파이썬의 데이터 타입
1. **Number(숫자)**: 정수, 실수, 8진수, 16진수
   - 파이썬은 다른 언어와 달리 정수나 실수를 딱히 나누지 않습니다. 하나의 숫자라고 생각할 수 있습니다.

2. **String(문자열)**: 문자와 문자열(문자의 배열)
   - 파이썬에서는 다른 언어와 달리 문자와 문자열을 구분하지 않습니다.

3. **List**: 다양한 타입의 리터럴 저장, 가변, 중복 허용
   - ‘[’ ‘]’ 기호를 사용합니다.

4. **Tuple**: 다양한 타입의 리터럴 저장, 불변, 중복 허용
   - ‘(’ ‘)’ 기호를 사용합니다.

5. **Dictionary**: Key:Value의 형태로 저장, Key 중복 불허, Value는 중복 허용

6. **Set**: 다양한 타입의 리터럴 저장, 가변, 중복 불허

7. **Boolean**: True 또는 False 저장

8. **Class**: 데이터 + 저장

[dataTypes.py](./dataTypes.py) 코드 예시:
```python
# 출력함수
def p(value):
    print(value, '\n')

# 숫자형
p(1)
p(0.1)
p(3E5)
p(0o10)
p(0xFF)
p(1+2)
p(3**3)
p(7%4)
p(7//4)

# 문자형
p('Hello')
p("Hello")
p('''
Hello
There
''')
p("Hello 'tom'")
p('Hello "tom"')
p("Hello \"tom\"")
p("Hello\ntom")
p("Hello" + "tom")
p("Hello"*3)
p(len("Hello"))

str = "Hello There"
p(str[0])
p(str[:3])
p(str[3:])
p(str[3:5])
p(str[::2])

str = "jane"
num = 20
p('%s은 %d살 입니다' % (str, num))
p('{0}은 {1}살 입니다'.format(str, num))
p(f'{str}은 {num}살 입니다')
p("%10s" % "hello")
p("%10.4f" % 3.141592)

str = "hello there"
p(str.count('e'))
p(str.find('e'))
p(str.find('k'))
p(str.index('e'))
# p(str.index('k'))
p(str.upper().lower())
p(str.replace("e", "k"))
p(str.split())
p(str.split("e"))

# List
even = [2, 4, 6, 8]
odd = [1, 3, 5, 7]

p(even)
p(odd)

p(even[0])
p(even[0:])
p(even[:2])
p(even[0::2])
p(even[-1])

p(even+odd)
p(even*3)

p(even.index(6))

p(len(even))

del even[0]
p(even)

even.append(10)
p(even)

even.sort()
p(even)

even.reverse()
p(even)

even.insert(0, 2)
p(even)

even.remove(2)
p(even)

even.pop()
p(even)

even.extend([4, 2])
p(even)

# Tuple
# Tuple은 List와 같지만 값의 변경 및 삭제가 불가 (오류발생)
# List는 [], Tuple은 () 문법 사용

# Dictionary
# Key:Value(item)으로 데이터를 저장
# Key는 item을 구별하는 식별자의 역할로 중복불허
# Value는 Key를 통해 검색되는 값으로 중복허용
# {key:value} 문법 사용

dic = {"name": "홍길동", "age": 20, "address": "서울"}
p(dic)
p(dic["name"])
p(dic.get("name"))

dic["gender"] = "남"
p(dic)

del dic["address"]
p(dic)

p(dic.keys())
p(dic.values())
p(list(dic.keys()))
p(list(dic.values()))

p(dic.items())

dic.clear()
p(dic)

p('address' in dic)

# Set
# 집합은 중복된 값을 허용하지 않는 자료구조
# 문법은 {} 사용

s = {1, 2, 3, 4, 5, 1, 2, 3}
p(s)

s = set(['a', 'b', 'c'])
p(s)

s = set(['hello'])
p(s)

s1 = {1, 2, 3, 4, 5}
s2 = {3, 4, 5, 6, 7}
p(s1 & s2)
p(s1 | s2)
p(s1 - s2)
p(s2 - s1)

# Boolean
# True 또는 False의 값을 저장
# 문자열에 문자가 1개이상 있으면 True
# 리스트, 튜플에 요소가 1개이상 있으면 True
# 딕셔너리에 item이 1개이상 있으면 True
# 0은 False, 0이 아니면 True
# None은 False

print(1==1)
print(2<1)
print(bool("hello"))
print(bool(""))
```

## logic
모든 알고리즘은 “분기”와 “반복”의 조합으로 이루어집니다.

분기
- if문

반복
- for

- for in

- while

``` python
# p 함수
def p(value):
    print(f"{value}\n")

# if

a = 10
if a > 5:
    p("a는 5보다 큼")
else:
    p("a는 5보다 크지 않음")

a = 10
if a < 5:
    p("a는 5보다 작음")
elif a < 10:
    p("a는 5보다 작지 않지만 10보다 작음")
else:
    p("a는 10보다 작지 않음")

# for

for i in range(1, 10, 2):
    p(i)

l = [1, 2, 3, 4, 5]
for i in l:
    p(i)

l = {'a': 1, 'b': 2, 'c': 3}
for i in l.values():
    p(i)

l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in l:
    if i % 2 == 0:
        continue
    if i == 9:
        break
    else:
        p(i)

# while
# 조건이 True인 동안 무한반복
# 반드시 반복조건이 False인 경우가 있어야 무한루프하지 않음

a = 0
while a < 10:
    p(a)
    a = a + 1

a = 0
while a < 10:
    if a == 5:
        break
    p(a)
    a = a + 1

```


