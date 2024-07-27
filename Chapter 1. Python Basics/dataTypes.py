# Output Function
def p(value):
    print(value, '\n')


# Numeric Types

p(1)
p(0.1)
p(3E5)
p(0o10)
p(0xFF)
p(1+2)
p(3**3)
p(7%4)
p(7//4)


# String Types

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
p('%s is %d years old' % (str, num))
p('{0} is {1} years old'.format(str, num))
p(f'{str} is {num} years old')
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
# No size limit and can store any type
# Elements are accessed by their index

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
# Tuples are like lists but cannot be modified (immutable)
# Lists use [], Tuples use ()


# Dictionary
# Stores data as Key:Value pairs
# Keys are unique identifiers and cannot be duplicated
# Values can be duplicated
# Uses {key:value} syntax

dic = {"name": "Hong Gil-dong", "age": 20, "address": "Seoul"}
p(dic)
p(dic["name"])
p(dic.get("name"))

dic["gender"] = "male"
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
# Sets are data structures that do not allow duplicate values
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
# Lists or tuples with one or more elements are True
# Dictionaries with one or more items are True
# 0 is False, non-zero is True
# None is False

print(1==1)
print(2<1)
print(bool("hello"))
print(bool(""))
