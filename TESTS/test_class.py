from pathlib import Path
from typing import List, Dict, Any
import json


class Test:
    def __init__(self, source_dir: str = "G", output_dir: str = "output"):
        self.source_directory = Path(source_dir)
        self.output_directory = Path(output_dir)

        if not self.source_directory.exists():
            raise ValueError(f"Source directory not found: {self.source_directory}")
        
        # Check read permission using pathlib / stat or try accessing it
        try:
            next(self.source_directory.iterdir())
        except PermissionError:
            raise ValueError(f"Source directory does not have read permission: {self.source_directory}")
        except StopIteration:
            pass  # Directory is empty, which is fine

    def _find_files(self) -> List[Path]:
        """Find all supported source files."""
        supported_extensions = {".py", ".md", ".txt"}

        return [
            file_path
            for file_path in self.source_directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix in supported_extensions
        ]
    
    def _save_chunks(self, file_paths: List[Path]) -> None:
        """Convert file paths to chunk metadata and save to disk."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / "chunks.json"

        # Convert Path objects into JSON-serializable dictionaries
        chunks = [
            {
                "path": str(p),
                "name": p.name,
                "extension": p.suffix
            }
            for p in file_paths
        ]
        print("chunks", chunks)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4)
    
if __name__ == "__main__":
    t = Test()
    res = t._find_files()
    print(res)
    t._save_chunks(res)