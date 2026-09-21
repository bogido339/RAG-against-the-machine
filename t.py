from pathlib import Path

p = Path("/home/user") / "projects" / "demo.txt"
print(p.name)
print(p.parent)
print(p.suffix)

