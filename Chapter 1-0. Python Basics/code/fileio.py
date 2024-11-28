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
