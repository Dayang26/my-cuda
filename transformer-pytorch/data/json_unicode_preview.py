import json

path = "./dataset/test.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(json.dumps(data[:5], ensure_ascii=False, indent=2))

print(type(data), len(data), type(data[0]), len(data[0]))
