# 1. Create the modules directory and then create the __init__.py file.
# 2. Create the calc directory under the modules directory and then create the __init__.py file.
# 3. Create the calc.py file in the modules/calc directory.
# def add(a, b):
#     return a + b
# def multi(a, b):
#     return a * b

import modules.calc.calc

result1 = modules.calc.calc.add(3, 5)
print(result1)

result2 = modules.calc.calc.multi(3, 5)
print(result2)

from modules.calc.calc import add
print(add(5, 5))
