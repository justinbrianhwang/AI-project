### objectdata.py

## pickle
import pickle
obj = {
    "name": "Hong Gil-dong",
    "age": 20
}
# Write the data of obj object to obj.obj file as binary
# wb: binary write mode
with open('obj.obj', 'wb') as f:
    pickle.dump(obj, f)
# Read binary data from obj.obj file
# rb: binary read mode
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
