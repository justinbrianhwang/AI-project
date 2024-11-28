# Function

def add(a, b):
    return a + b

print(add(3,4))

def multi(a, b=3):
    return a * b
print(multi(3))

c = add(3,3)
d = multi(3, 3)
print(c, d)

def func1(a, b):
    result1 = a + b
# print(result1)

lambda_add = lambda a, b: a+b
print(lambda_add(3,3))
