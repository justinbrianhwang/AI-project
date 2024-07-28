### objectdata.py

## pickle
import pickle
obj = {
    "name": "홍길동",
    "age": 20
}
# obj.obj파일에 obj객체의 데이터를 바이너리로 쓰기
# wb : 바이너리 쓰기 모드
with open('obj.obj', 'wb') as f:
    pickle.dump(obj, f)
# obj.obj파일에서 바이너리 데이터를 읽기
# rb : 바이너리 읽기 모드
with open('obj.obj', 'rb') as f:
    print(pickle.load(f))


## shelve
import shelve
def save(key, value):
    with shelve.open("shelve") as f:
        f[key] = value
def get(key):
    with shelve.open("shelve") as f:
        return f[key]

save("number", [1, 2, 3, 4, 5])
save("string", ["a", "b", "c"])
print(get("number"))
print(get("string"))










