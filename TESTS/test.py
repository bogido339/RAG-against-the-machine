import hashlib
from pathlib import Path


def get_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


file = Path("example.py")

old_hash = get_file_hash(file)
print(old_hash)

# After some time...
new_hash = get_file_hash(file)

if old_hash != new_hash:
    print("File content changed!")
else:
    print("File content has not changed.")