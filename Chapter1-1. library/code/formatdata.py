### formatdata.py

## Format Data: The format of data exchanged over a network
# Data: Text (CSV, XML, JSON ...), Binary (jpg, mp4 ...)
# Network data
# 1. CSV (Comma Separated Value)
# 2. XML (Extensible Markup Language)
#    Advantages: Data structure + data, Disadvantages: Uses many bytes to represent data => High network cost
# 3. JSON (JavaScript Object Notation)
#    Advantages: Reduces network cost, Disadvantages: Cannot represent data structures
# 4. XML + JSON

# CSV
import csv
with open('csvdata.csv', mode='w', encoding='utf-8') as f:
    # delimiter: data separator
    # quotechar: character recognized as a string
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
