### formatdata.py

## 포맷데이터 : 네트워크상에서 주고 받는 데이터의 형식
# 데이터 : 문자(CSV, XML, JSON ...), 바이너리(jpg,mp4 ...)
# 네트워크상의 데이터
# 1. CSV (Comma Separated Value, 콤마로 구분된 값)
# 2. XML (Extensible Markup Language, 확장가능한 표기 언어)
#    장점:데이터구조+데이터, 단점:데이터 표현에 많은 바이트 사용 => 네트워크 비용 크다
# 3. JSON (JavaScript Object Notation, 자바스크립트 객체표기법)
#    장점:네트워크 비용이 절감, 단점:데이터 구조표현이 불가
# 4. XML + JSON

# CSV
import csv
with open('csvdata.csv', mode='w', encoding='utf-8') as f:
    # delimiter : 데이터 구분자
    # quotechar : 문자열로 인식하는 문자
    writer = csv.writer(f, delimiter=',', quotechar="'")
    # csv파일에 행 쓰기
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














