import json

charge2id = {}
with open("charge2id.json") as f:
    charge2id = json.load(f)

id2charge = {}

for k in charge2id.keys():
    id2charge[charge2id[k]] = k

with open("id2charge.json", "w") as f:
    json.dump(id2charge, f, ensure_ascii=False)
