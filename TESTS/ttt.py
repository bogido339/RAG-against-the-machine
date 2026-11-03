from pathlib import Path

# Create a path pointing to a file or directory
path = Path("documents/report.txt")

# Get special system locations
cwd = Path.cwd()       # Current working directory
home = Path.home()     # User's home directory

print(type(cwd))
print(home)

print(type(pass))