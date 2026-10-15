from pathlib import Path
import os

def test(source_directory):
    supported_extensions = [".py", ".md"]

    found_files = []

    if not source_directory.exists():
        raise ValueError(f"Source directory not found: {source_directory}")
    
    if not source_directory.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {source_directory}"
        )

    if not os.access(source_directory, os.R_OK):
        raise PermissionError(
            f"Source directory does not have read permission: "
            f"{source_directory}"
        )

    for file_path in source_directory.rglob("*"):

        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            found_files.append(file_path)

    return found_files


print(test(Path("src/cli.py")))