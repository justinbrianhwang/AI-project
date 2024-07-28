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

[dataTypes.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/dataTypes.py) code example:
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


[logic.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/logic.py)
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


# Python Basic Concepts Summary

### Function

A function is a unit of a program designed to execute blocks of code when called.

- Function Declaration

```python
def functionName(parameterList):
    code...
    return value
```

- Function Call

```python
functionName(argument)
variable = function(argument)
```

- Variables passed to a function are called arguments, and variables used to execute code within the function are called parameters.
- Structuring a function to handle only one functionality is efficient for reuse and maintenance.

[function.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/function.py)
```python
# Functions

def add(a, b):
    return a + b

print(add(3,4))  # 7

def multi(a, b=3):
    return a * b

print(multi(3))  # 9
print(multi(3, 4))  # 12

c = add(3,3)
d = multi(3, 3)
print(c, d)  # 6, 9

def func1(a, b):
    result1 = a + b
# print(result1)

lambda_add = lambda a, b: a+b
print(lambda_add(3,3))  # 6
```

### I/O

Input, Processing, Output. Programming involves processing input data and converting it into output data.

- Input
    - keyboard
        - `input()`: Stores user-entered characters as a string.
    - file
        - `open(filename, mode)`: Modes are w(write), r(read), a(append).
        - `readline()`: Reads a line from the file and stores it as a string.
        - `readlines()`: Reads all lines from the file and stores them as a list.
        - `read()`: Stores the entire file content as a string.
        - `write()`: Writes a string passed as an argument to the file.
        - `close()`: Closes the used file.
- Output
    - monitor
        - `print()`: Outputs the characters passed as an argument to the console.

[fileio.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/fileio.py)
```python
# File I/O

f = open("datafile.txt", "w")
f.write("Hello")
f.close()

f = open("datafile.txt", "a")
f.write(" There")
f.close()

f = open("datafile.txt", "r")
print(f.read())
f.close()

jsonObj = '{"name":"홍길동","age":20,"address":"서울"}'

f = open("jsonObj.json", "w")
f.write(jsonObj)
f.close()

f = open("jsonObj.json", "r")
jsonStr = f.read()
print(jsonStr)
f.close()
```

### Class & Object

A class is a blueprint for creating objects, and an object is a programming unit that bundles data and functions.

- Class
    - Class Declaration
    
    ```python
    class className:
        data..
        function...
    ```
    
- Object
    - Object Creation
    
    ```python
    object = ClassName()
    ```
    
- `self`: A keyword that refers to the object itself created in memory (equivalent to `this` in other languages).
- Constructor
    - A special method for assigning data to the object (uses `__init__` syntax).
    - All methods are functions, and a function accessed through an object is called a method.
- Inheritance
    - Enhances reusability by inheriting an existing class.
    - Syntax: `class className(ParentClass)`
- Method Overriding
    - When two classes are in an inheritance relationship, the subclass redefines the superclass's method.
    - A polymorphism technique that selects the same form method based on the object type.
- Class Variable
    - Used when all objects created through the class need the same value.
    - Declared at the bottom of the class declaration.

☆ Three Main Concepts of Object-Oriented Programming

1. Inheritance
    - Reusing what has been created before.
2. Polymorphism
    - Having different properties while maintaining the same form.
    - Example: Overriding.
3. Abstraction
    - Simplifying as much as possible without losing the essence.

[class&object.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/class%26object.py)
```python
# Class

class Human:
    humanCount = 0
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def setAge(self, age):
        self.age = age
    def getAge(self):
        return self.age

# Object

hong = Human("홍길동", 20)
kang = Human("강감찬", 30)
print(hong.name)  # 홍길동
print(kang.age)  # 30

kang.setAge(40)
print(kang.getAge())  # 40

Human.humanCount = 1
print(kang.humanCount)  # 1

# Inheritance & Overriding

class Vehicle:
    def __init__(self, name, tireCount):
        self.name = name
        self.tireCount = tireCount
    def getName(self):
        return f"This vehicle is {self.name}"

class Car(Vehicle):
    def getName(self):
        return f"This car is {self.name}"

car = Car("bentz", 4)
print(car.name, car.tireCount)  # bentz 4

car = Vehicle("bmw", 4)
print(car.getName())  # This vehicle is bmw

car = Car("honda", 4)
print(car.getName())  # This car is honda
```

### Module & Package

Modularizing/packaging numerous data and logic is crucial for program scalability and management.

- Module
    - A programming unit that gathers functions and variables (typically a file).
    - To import a module, use the `import` statement: `import module_name`.
    - To import specific items from a module: `from module_name import item_name`.
- `__name__=="__main__"`: Checks if it is the main file, not a module.
- Package
    - A programming unit for managing modules hierarchically (typically a folder).
    - When a folder contains an `__init__.py` file, it is recognized as a package.
    - Modules can be used with `import package_name.module_name`.

[modlue&package.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/module%26package.py)
```python
# 1. Create the modules directory and then create the __init__.py file.
# 2. Create the calc directory under the modules directory and then create the __init__.py file.
# 3. Create the calc.py file in the modules/calc directory.

# calc.py
def add(a, b):
    return a + b
def multi(a, b):
    return a * b

import modules.calc.calc

result1 = modules.calc.calc.add(3, 5)
print(result1)  # 8

result2 = modules.calc.calc.multi(3, 5)
print(result2)  # 15

from modules.calc.calc import add
print(add(5, 5))  # 10
```

### Exception Handling

Exceptions are errors that can occur at runtime.
Handling exceptions is an important programming issue to enhance the stability of the developed program.

- Exception Handling Syntax
    - `try`: Block of code where an exception might occur.
    - `except`: Block of code to handle the exception.
    - `finally`: Block of code to execute regardless of whether an exception occurs.
- `raise`
    - Keyword to deliberately raise an exception.
- Creating Custom Exceptions
    - Inherit from the `Exception` class and override the `__str__()` method.

[exceptionHandling.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/exceptionHandling.py)
```python
# Exception Handling

try:
    result = 10/0
except ZeroDivisionError:
    print("Division by zero exception occurred!")
finally:
    print("Executed regardless of exception!")

class Under19Exception(Exception):
    def __str__(self):
        return "Under 19 Exception occurred!"

age = 18
if age < 19:
    try:
        raise Under19Exception
    except Under19Exception as e:
        print(e)
    finally:
        print("Exception handling complete!")
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

[dataTypes.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/dataTypes.py) 코드 예시:
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

# Python 기초 개념 정리

### Function

함수는 호출하여 함수 블럭 내의 코드들을 실행하기 위한 프로그램의 기능 단위

- 함수의 선언

```python
def functionName(parameterList):
    code...
    return value
```

- 함수의 호출

```python
functionName(argument)
variable = function(argument)
```

- 함수에 전달되는 값을 가지는 변수를 인자라고 하고, 함수 내 코드들을 실행하기 위한 변수들을 파라미터라 한다.
- 하나의 함수는 하나의 기능만을 처리하도록 구조화하는 것이 재사용과 유지관리 차원에서 효율적

[function.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/function.py)
```python
# 함수

def add(a, b):
    return a + b

print(add(3,4))  # 7

def multi(a, b=3):
    return a * b

print(multi(3))  # 9
print(multi(3, 4))  # 12

c = add(3,3)
d = multi(3, 3)
print(c, d)  # 6, 9

def func1(a, b):
    result1 = a + b
# print(result1)

lambda_add = lambda a, b: a+b
print(lambda_add(3,3))  # 6
```

### I/O

Input, Processing, Output. 프로그래밍은 입력 데이터를 처리하여 출력 데이터화 하는 작업이다.

- Input
    - keyboard
        - `input()`: 사용자가 입력한 문자들은 문자열로 저장
    - file
        - `open(파일명, 모드)`: 모드는 w(읽고 쓰기), r(읽기), a(추가)
        - `readline()`: 파일로부터 한줄을 읽어서 문자열로 저장
        - `readlines()`: 파일로부터 모든 줄을 읽어서 List로 저장
        - `read()`: 파일 전체의 내용을 문자열로 저장
        - `write()`: 인자로 전달받은 문자열을 파일에 쓰기
        - `close()`: 사용한 파일을 닫기
- Output
    - monitor
        - `print()`: 인자로 전달받은 문자들을 콘솔에 출력

[fileio.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/fileio.py)
```python
# file io

f = open("datafile.txt", "w")
f.write("Hello")
f.close()

f = open("datafile.txt", "a")
f.write(" There")
f.close()

f = open("datafile.txt", "r")
print(f.read())
f.close()

jsonObj = '{"name":"홍길동","age":20,"address":"서울"}'

f = open("jsonObj.json", "w")
f.write(jsonObj)
f.close()

f = open("jsonObj.json", "r")
jsonStr = f.read()
print(jsonStr)
f.close()
```

### Class & Object

Class는 Object를 생성하기 위한 틀, Object는 데이터와 기능을 묶은 프로그래밍 단위

- Class
    - 클래스 선언
    
    ```python
    class className:
        data..
        function...
    ```
    
- Object
    - 객체 생성
    
    ```python
    object = ClassName()
    ```
    
- `self`: 메모리에 생성된 객체 자신을 가리키는 키워드 (타 언어에서는 this)
- Constructor
    - 객체의 데이터를 할당하기 위한 특별한 메서드 (`__init__` 문법 사용)
    - 메서드는 모두 함수이며, 객체로 접근하는 함수를 메서드라 함
- 상속
    - 이미 만들어져 있는 클래스를 상속 받아 사용함으로써 재사용성 증진
    - 문법: `class className(ParentClass)`
- 메서드 오버라이딩
    - 두 클래스가 상속관계에 있을 때 상위클래스의 메서드를 하위클래스의 메서드가 재정의
    - 동일한 형태의 메서드를 호출하는 객체의 타입에 따라 선택하게 하는 객체지향 기법
- 클래스 변수
    - 클래스를 통해 생성되는 모든 객체들에 동일한 값이 필요할 때 사용
    - 클래스 선언 하단에 클래스 변수를 선언

☆ 객체 지향의 3대 개념

1. 상속
    - 기본에 만들어진 것 재사용
2. 다형성
    - 동일한 형태인데 다른 성질을 갖도록
    - 예) 오버라이딩
3. 추상화
    - 본연의 성질을 잃지 않는 선에서 최대한 단순화

[class&object.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/class%26object.py)
```python
# class

class Human:
    humanCount = 0
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def setAge(self, age):
        self.age = age
    def getAge(self):
        return self.age

# object

hong = Human("홍길동", 20)
kang = Human("강감찬", 30)
print(hong.name)  # 홍길동
print(kang.age)  # 30

kang.setAge(40)
print(kang.getAge())  # 40

Human.humanCount = 1
print(kang.humanCount)  # 1

# 상속 & 오버라이딩

class Vehicle:
    def __init__(self, name, tireCount):
        self.name = name
        self.tireCount = tireCount
    def getName(self):
        return f"이 탈것은 {self.name} 입니다"

class Car(Vehicle):
    def getName(self):
        return f"이 차는 {self.name} 입니다"

car = Car("bentz", 4)
print(car.name, car.tireCount)  # bentz 4

car = Vehicle("bmw", 4)
print(car.getName())  # 이 탈것은 bmw 입니다

car = Car("honda", 4)
print(car.getName())  # 이 차는 honda 입니다
```

### Module & Package

수 많은 데이터와 로직들을 모듈화/패키지화 하는 것은 프로그램의 확장성과 관리 측면에서 매우 중요

- 모듈
    - 함수나 변수를 모아놓은 프로그래밍 단위(일반적으로 파일)
    - 모듈을 불러올 때는 `import` 구문 사용: `import 모듈명`
    - 모듈에서 특정한 것들만 불러올 때는 : `from 모듈명 import 불러올 것들`
- `__name__=="__main__"`: 모듈이 아닌 메인 파일인지 검사
- 패키지
    - 모듈을 계층적으로 관리하기 위한 프로그래밍 단위 (일반적으로 폴더)
    - 폴더 내에 아무 내용이 없는 `__init__.py` 파일을 작성하면 패키지로 인식
    - `import 패키지명.모듈명`으로 모듈을 호출하여 사용

[modlue&package.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/module%26package.py)
```python
# 1. modules 디렉토리 생성 후 __init__.py 파일 생성
# 2. modules 디렉토리 하위에 calc 디렉토리 생성 후 __init__.py 파일 생성
# 3. modules/calc 디렉토리에 calc.py 파일 생성
# def add(a, b):
#     return a + b
# def multi(a, b):
#     return a * b

import modules.calc.calc

result1 = modules.calc.calc.add(3, 5)
print(result1)  # 8

result2 = modules.calc.calc.multi(3, 5)
print(result2)  # 15

from modules.calc.calc import add
print(add(5, 5))  # 10
```

### Exception Handling

예외한 프로그램 실행시점에 발생할 수 있는 에러
예외처리는 개발한 프로그램의 안정성을 높이는 중요한 프로그래밍 이슈

- 예외처리 문법
    - `try`: 예외 발생 가능한 코드 블록
    - `except`: 예외를 처리하기 위한 코드 블록
    - `finally`: 예외 발생과 상관없이 수행할 코드 블록
- `raise`
    - 예외를 일부러 발생시키는 키워드
- 사용자 정의 예외 만들기
    - `Exception` 클래스를 상속받아 `__str__()` 메서드를 오버라이딩


[exceptionHandling.py](https://github.com/justinbrianhwang/AI-project/blob/main/Chapter%201.%20Python%20Basics/code/exceptionHandling.py)
```python
# exception handling

try:
    result = 10/0
except ZeroDivisionError:
    print("0으로 나누기 예외 발생!")
finally:
    print("예외 발생과 상관없이 수행!")

class Under19Exception(Exception):
    def __str__(self):
        return "19세 이하 관람불가 예외 발생!"

age = 18
if age < 19:
    try:
        raise Under19Exception
    except Under19Exception as e:
        print(e)
    finally:
        print("예외 처리 완료!")
```



