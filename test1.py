import json


chunks_path = "exemple_falder/file.json"

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)
    print(type(chunks))