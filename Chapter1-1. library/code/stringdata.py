# 문자열 처리 라이브러리

# 내장라이브러리는 파이썬에서 기본 제공하는 라이브러리, 별도 설치 불필요
# 외부라이브러리는 파이썬 설치 후에 별도로 설치해야 사용 가능
# 라이브러리 사용시는 import 구문 사용

# 출력함수
def p(str):
    print(str, "\n")

## textwrap
import textwrap

str = "Hello Python"
# 문자열 길이 축약 (문자열, 길이, 대체문자열)
p(textwrap.shorten(str, width=10, placeholder="..."))

str = str * 10 # 문자열 10번 반복
p(str)
# 문자열 공백기준으로 11개 요소화 해서 리스트로 변환
wrapstr = textwrap.wrap(str, width=11)
p(wrapstr)

# 리스트의 각 요소들에 줄바꿈문자를 붙여서 문자열로 변환
p("\n".join(wrapstr))


## re (regular expression, 정규표현식)
# 문자열 전체에서 부분문자열들을 탐색, 추출, 대체하는데 사용되는
# 패턴문자열과 플래그문자열의 조합인 식
# 모든 프로그래밍 언어에서 공통적으로 사용되는 식이므로 학습 필수!

import re
str = "홍길동의 전화번호는 010-1234-5678"
pattern = re.compile("(\d{3})-(\d{4})-(\d{4})")
p(pattern.sub("\g<1> \g<2> \g<3>", str))

# 전화번호, 이메일, IP주소, 주민등록번호 등 자주 사용되는 패턴을 만들어 봅시다!

# 정규표현식 예제

# 1. 문자열에서 apple을 검색
text = "I like apple pie"
result = re.findall(r"apple", text)
p(result)

# 2. 1개 이상의 숫자를 검색
text = "My phone number is 010-1234-5678."
result = re.findall(r"\d+", text)
p(result)

# 3. 간단한 이메일주소 패턴 검색
text = "email address is example@email.com"
result = re.findall(r"\b\w+@\w+\.\w+", text)
p(result)

# 4. 간단한 휴대폰번호 패턴 검색
text = "Call me at 010-1234-5678"
result = re.findall(r"\d{3}-\d{4}-\d{4}", text)
p(result)

# 5. 영문대문자 패턴 검색
text = "Hello Python"
result = re.findall(r"[A-Z]", text)
print(result)

# 6. 문자열내의 불필요한 공백 제거
text = "Hello     Python     This is Me"
result = re.sub(r"\s+", " ", text)
p(result)

# 7. 문자열의 시작과 끝 검색
# ^:시작, [^]:부정  ex) ^a:a문자로 시작, [^a]:a문자가아님
text = "Hello World"
result = re.findall(r"^Hello|World$", text)
p(result)

# 8. 특정 단어로 시작하는 문자열 검색
# \b : word(\w, 알파벳대소문자 또는 숫자 또는 _)의 경계
text = "Start your journey with a smile. Start early to avoid traffic."
result = re.findall(r"\bStart\b[^.]*\.", text)
p(result)

# 9. 문자열에서 URL 검색
text = "Website URL is http://example.com or https://www.exampe.com"
result = re.findall(r"https?://[^\s]+", text)
p(result)

# 10. 날짜형식 (년도네자리-월두자리-일두자리) 검색
text = "오늘은 2024-07-27일 이고 내일은 2024-07-28일"
result = re.findall(r"\d{4}-\d{2}-\d{2}", text)
p(result)

















