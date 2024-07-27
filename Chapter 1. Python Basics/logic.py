# p 함수
def p(value):
    print(f"{value}\n")

# if

a = 10
if a>5:
    p("a는 5보다 큼")
else :
    p("a는 5보다 크지 않음")

a = 10
if a<5:
    p("a는 5보다 작음")
elif a<10 :
    p("a는 5보다 작지 않지만 10보다 작음")
else :
    p("a는 10보다 작지 않음")


# for

for i in range(1, 10, 2):
    p(i)

l = [1, 2, 3, 4, 5]
for i in l:
    p(i)

l = {'a':1, 'b':2, 'c':3}
for i in l.values():
    p(i)

l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in l:
    if i%2==0:
        continue
    if i==9:
        break;
    else:
        p(i)


# while
# 조건이 True인 동안 무한반복
# 반드시 반복조건이 False인 경우가 있어야 무한루프하지 않음

a = 0
while a<10:
    p(a)
    a = a + 1

a = 0
while a<10:
    if a==5:
        break
    p(a)
    a = a + 1

