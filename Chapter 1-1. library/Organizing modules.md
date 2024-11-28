# English
# Most Commonly Used Functions in Python Modules

## String Data

### `textwrap`
- `textwrap.wrap(text, width)`: Wraps the text into lines of the specified width, returning a list.
- `textwrap.fill(text, width)`: Wraps the text into lines of the specified width, returning a single string.
- `textwrap.shorten(text, width)`: Shortens the text to the specified width.
- `textwrap.indent(text, prefix)`: Adds the given prefix to the beginning of each line.
- `textwrap.dedent(text)`: Removes any common leading whitespace from every line.

### `re`
- `re.compile(pattern)`: Compiles a regular expression pattern into a regex object.
- `re.match(pattern, string)`: Checks for a match only at the beginning of the string.
- `re.search(pattern, string)`: Searches the string for a match to the pattern.
- `re.findall(pattern, string)`: Finds all matches of the pattern in the string, returning them as a list.
- `re.sub(pattern, repl, string)`: Replaces matches of the pattern in the string with a replacement string.

## Numeric Data

### `math`
- `math.sqrt(x)`: Returns the square root of x.
- `math.pow(x, y)`: Returns x raised to the power of y.
- `math.sin(x)`, `math.cos(x)`, `math.tan(x)`: Calculates the sine, cosine, and tangent of x.
- `math.log(x, base)`: Calculates the logarithm of x to the given base.
- `math.factorial(x)`: Returns the factorial of x.

### `decimal`
- `decimal.Decimal(value)`: Creates a Decimal object from the given value.
- `decimal.getcontext()`: Returns the current context.
- `decimal.setcontext(context)`: Sets the context.
- `decimal.localcontext(ctx=None)`: Creates a local context.

### `fractions`
- `fractions.Fraction(numerator, denominator)`: Creates a Fraction object.
- `fractions.Fraction.from_float(flt)`: Converts a floating-point number to a fraction.
- `fractions.Fraction.limit_denominator(max_denominator)`: Limits the denominator of the fraction.

### `random`
- `random.random()`: Returns a random float number between 0 and 1.
- `random.randint(a, b)`: Returns a random integer between a and b.
- `random.choice(seq)`: Returns a random element from the sequence.
- `random.shuffle(seq)`: Shuffles the sequence in place.
- `random.sample(population, k)`: Returns a sample of k elements from the population without replacement.

### `statistics`
- `statistics.mean(data)`: Calculates the mean of the data.
- `statistics.median(data)`: Calculates the median of the data.
- `statistics.mode(data)`: Calculates the mode of the data.
- `statistics.stdev(data)`: Calculates the standard deviation of the data.
- `statistics.variance(data)`: Calculates the variance of the data.

## Date Data

### `datetime`
- `datetime.datetime.now()`: Returns the current date and time.
- `datetime.datetime.strptime(date_string, format)`: Parses a string into a datetime object.
- `datetime.datetime.strftime(format)`: Formats a datetime object as a string.
- `datetime.timedelta(days=0, seconds=0, ...)`: Represents the difference between two datetime objects.

### `calendar`
- `calendar.month(year, month)`: Returns a string representing a month's calendar.
- `calendar.isleap(year)`: Checks if the year is a leap year.
- `calendar.leapdays(y1, y2)`: Returns the number of leap years in the range [y1, y2).
- `calendar.weekday(year, month, day)`: Returns the weekday of the specified date.

## Object Data

### `pickle`
- `pickle.dump(obj, file)`: Writes a pickled representation of the object to the file.
- `pickle.load(file)`: Loads a pickled object from the file.
- `pickle.dumps(obj)`: Returns the pickled representation of the object as a bytes object.
- `pickle.loads(bytes_object)`: Loads a pickled object from a bytes object.

### `shelve`
- `shelve.open(filename)`: Opens a persistent dictionary.
- `shelve.Shelf[key]`: Gets or sets an item in the shelf.
- `shelve.Shelf.sync()`: Writes any cached data to disk.
- `shelve.Shelf.close()`: Closes the shelf.

## Format Data

### `csv`
- `csv.reader(file)`: Reads a CSV file.
- `csv.writer(file)`: Writes to a CSV file.
- `csv.DictReader(file)`: Reads a CSV file into a dictionary.
- `csv.DictWriter(file, fieldnames)`: Writes a dictionary to a CSV file.

### `xml`
- `xml.etree.ElementTree.parse(file)`: Parses an XML file.
- `xml.etree.ElementTree.Element(tag)`: Creates a new XML element.
- `xml.etree.ElementTree.SubElement(parent, tag)`: Adds a subelement to the parent element.
- `xml.etree.ElementTree.tostring(element)`: Converts an XML element to a string.

### `json`
- `json.load(file)`: Reads JSON data from a file.
- `json.loads(string)`: Parses a JSON string.
- `json.dump(obj, file)`: Writes JSON data to a file.
- `json.dumps(obj)`: Converts an object to a JSON string.

## Remote JSON Data

### `requests`
- `requests.get(url)`: Sends a GET request.
- `requests.post(url, data)`: Sends a POST request.
- `requests.put(url, data)`: Sends a PUT request.
- `requests.delete(url)`: Sends a DELETE request.

### `urllib`
- `urllib.request.urlopen(url)`: Opens a URL.
- `urllib.parse.urlparse(url)`: Parses a URL.
- `urllib.parse.urlencode(query)`: Encodes a query string.
- `urllib.request.Request(url)`: Creates a request object.

### `aiohttp`
- `aiohttp.ClientSession()`: Creates a client session.
- `aiohttp.ClientSession.get(url)`: Sends an asynchronous GET request.
- `aiohttp.ClientSession.post(url, data)`: Sends an asynchronous POST request.
- `aiohttp.ClientSession.put(url, data)`: Sends an asynchronous PUT request.
- `aiohttp.ClientSession.delete(url)`: Sends an asynchronous DELETE request.







# Korean
# Python 모듈에서 가장 많이 사용되는 함수들

## 문자열 데이터

### `textwrap`
- `textwrap.wrap(text, width)`: 주어진 너비(width)로 텍스트를 여러 줄로 나누어 리스트로 반환합니다.
- `textwrap.fill(text, width)`: 주어진 너비로 텍스트를 여러 줄로 나누어 하나의 문자열로 반환합니다.
- `textwrap.shorten(text, width)`: 텍스트를 주어진 너비로 줄입니다.
- `textwrap.indent(text, prefix)`: 각 줄의 시작에 접두사를 추가합니다.
- `textwrap.dedent(text)`: 각 줄에서 공통된 공백을 제거합니다.

### `re`
- `re.compile(pattern)`: 정규 표현식을 컴파일합니다.
- `re.match(pattern, string)`: 문자열의 시작에서 정규 표현식과 매치되는지 검사합니다.
- `re.search(pattern, string)`: 문자열 전체에서 정규 표현식과 매치되는 부분을 검색합니다.
- `re.findall(pattern, string)`: 정규 표현식과 매치되는 모든 부분을 리스트로 반환합니다.
- `re.sub(pattern, repl, string)`: 정규 표현식과 매치되는 부분을 다른 문자열로 치환합니다.

## 숫자 데이터

### `math`
- `math.sqrt(x)`: x의 제곱근을 반환합니다.
- `math.pow(x, y)`: x의 y 제곱을 반환합니다.
- `math.sin(x)`, `math.cos(x)`, `math.tan(x)`: 삼각함수를 계산합니다.
- `math.log(x, base)`: 로그를 계산합니다.
- `math.factorial(x)`: x의 팩토리얼을 계산합니다.

### `decimal`
- `decimal.Decimal(value)`: Decimal 객체를 생성합니다.
- `decimal.getcontext()`: 현재 문맥(Context) 객체를 반환합니다.
- `decimal.setcontext(context)`: 문맥을 설정합니다.
- `decimal.localcontext(ctx=None)`: 지역 문맥을 설정합니다.

### `fractions`
- `fractions.Fraction(numerator, denominator)`: 분수 객체를 생성합니다.
- `fractions.Fraction.from_float(flt)`: 부동 소수점 숫자를 분수로 변환합니다.
- `fractions.Fraction.limit_denominator(max_denominator)`: 최대 분모를 가지는 분수로 변환합니다.

### `random`
- `random.random()`: 0과 1 사이의 난수를 반환합니다.
- `random.randint(a, b)`: [a, b] 범위의 정수를 반환합니다.
- `random.choice(seq)`: 시퀀스에서 임의의 요소를 반환합니다.
- `random.shuffle(seq)`: 시퀀스를 섞습니다.
- `random.sample(population, k)`: 모집단에서 k개의 요소를 비복원 추출합니다.

### `statistics`
- `statistics.mean(data)`: 평균을 계산합니다.
- `statistics.median(data)`: 중앙값을 계산합니다.
- `statistics.mode(data)`: 최빈값을 계산합니다.
- `statistics.stdev(data)`: 표준 편차를 계산합니다.
- `statistics.variance(data)`: 분산을 계산합니다.

## 날짜 데이터

### `datetime`
- `datetime.datetime.now()`: 현재 날짜와 시간을 반환합니다.
- `datetime.datetime.strptime(date_string, format)`: 문자열을 datetime 객체로 변환합니다.
- `datetime.datetime.strftime(format)`: datetime 객체를 문자열로 변환합니다.
- `datetime.timedelta(days=0, seconds=0, ...)`: 두 datetime 간의 차이를 나타냅니다.

### `calendar`
- `calendar.month(year, month)`: 특정 년도와 월의 달력을 반환합니다.
- `calendar.isleap(year)`: 윤년인지 확인합니다.
- `calendar.leapdays(y1, y2)`: 두 연도 사이의 윤년 수를 계산합니다.
- `calendar.weekday(year, month, day)`: 특정 날짜의 요일을 반환합니다.

## 객체 데이터

### `pickle`
- `pickle.dump(obj, file)`: 객체를 파일에 저장합니다.
- `pickle.load(file)`: 파일에서 객체를 읽어옵니다.
- `pickle.dumps(obj)`: 객체를 바이트 스트림으로 변환합니다.
- `pickle.loads(bytes_object)`: 바이트 스트림을 객체로 변환합니다.

### `shelve`
- `shelve.open(filename)`: 새로운 선반 객체를 엽니다.
- `shelve.Shelf[key]`: 선반에서 항목을 가져오거나 설정합니다.
- `shelve.Shelf.sync()`: 데이터베이스의 모든 항목을 디스크에 씁니다.
- `shelve.Shelf.close()`: 선반을 닫습니다.

## 포맷 데이터

### `csv`
- `csv.reader(file)`: CSV 파일을 읽습니다.
- `csv.writer(file)`: CSV 파일에 씁니다.
- `csv.DictReader(file)`: CSV 파일을 딕셔너리 형식으로 읽습니다.
- `csv.DictWriter(file, fieldnames)`: CSV 파일에 딕셔너리 형식으로 씁니다.

### `xml`
- `xml.etree.ElementTree.parse(file)`: XML 파일을 파싱합니다.
- `xml.etree.ElementTree.Element(tag)`: 새로운 XML 요소를 생성합니다.
- `xml.etree.ElementTree.SubElement(parent, tag)`: 부모 요소에 하위 요소를 추가합니다.
- `xml.etree.ElementTree.tostring(element)`: XML 요소를 문자열로 변환합니다.

### `json`
- `json.load(file)`: 파일에서 JSON 데이터를 읽습니다.
- `json.loads(string)`: 문자열에서 JSON 데이터를 읽습니다.
- `json.dump(obj, file)`: 파일에 JSON 데이터를 씁니다.
- `json.dumps(obj)`: 문자열로 JSON 데이터를 변환합니다.

## 원격 json 데이터

### `requests`
- `requests.get(url)`: GET 요청을 보냅니다.
- `requests.post(url, data)`: POST 요청을 보냅니다.
- `requests.put(url, data)`: PUT 요청을 보냅니다.
- `requests.delete(url)`: DELETE 요청을 보냅니다.

### `urllib`
- `urllib.request.urlopen(url)`: URL을 엽니다.
- `urllib.parse.urlparse(url)`: URL을 파싱합니다.
- `urllib.parse.urlencode(query)`: 쿼리 문자열을 인코딩합니다.
- `urllib.request.Request(url)`: 요청 객체를 생성합니다.

### `aiohttp`
- `aiohttp.ClientSession()`: 클라이언트 세션을 생성합니다.
- `aiohttp.ClientSession.get(url)`: 비동기 GET 요청을 보냅니다.
- `aiohttp.ClientSession.post(url, data)`: 비동기 POST 요청을 보냅니다.
- `aiohttp.ClientSession.put(url, data)`: 비동기 PUT 요청을 보냅니다.
- `aiohttp.ClientSession.delete(url)`: 비동기 DELETE 요청을 보냅니다.
