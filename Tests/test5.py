from pathlib import Path

# 1. Define the path to your file
file_path = Path("src/main.py")

# 2. Read the entire content as a string
content = file_path.read_text(encoding="utf-8")

print(content)
