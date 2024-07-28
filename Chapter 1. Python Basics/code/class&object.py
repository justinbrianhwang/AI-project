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

hong = Human("Hong Gil-dong", 20)
kang = Human("Gang Gam-chan", 30)
print(hong.name)
print(kang.age)

kang.setAge(40)
print(kang.getAge())

Human.humanCount = 1
print(kang.humanCount)


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

car = Car("Benz", 4)
print(car.name, car.tireCount)

car = Vehicle("BMW", 4)
print(car.getName())

car = Car("Honda", 4)
print(car.getName())
