from typing import List, Dict, Any
from pathlib import Path


class BaseIndexer():
    """Abstract base class defining the indexing pipeline."""

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def chunk_markdown(self, file_content: str) -> List[Dict[str, Any]]:
        """Process and chunk markdown content based on structural boundaries."""
        pass

    def chunk_text(self, file_content: str) -> List[Dict[str, Any]]:
        pass

    def chunk_python(self, file_content: str) -> List[Dict[str, Any]]:
        """Process and chunk Python source code (e.g., by classes/functions)."""
        pass

    def build_index(self, raw_data_dir: str, processed_data_dir: str) -> None:
        """Traverse the corpus, apply the correct chunking strategy, and save the index."""
        folder = Path("resources/vllm-0.10.1")

        for file in folder.rglob("*"):
            if file.suffix == ".py":
                self.chunk_python(file)
            elif file.suffix == ".md":
                self.chunk_markdown(file)
            elif file.suffix == ".txt":
                self.chunk_text(file)
