from typing import List, Dict, Any
from pathlib import Path
import json

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
import bm25s


class IndexerError(Exception):
    ...


class Indexer:
    """Build the searchable index from the source codebase."""

    def __init__(self, max_chunk_size: int = 2000):

        if not isinstance(max_chunk_size, int) or max_chunk_size < 1:
            raise IndexerError("just positif intejer > 0")

        self.max_chunk_size = max_chunk_size
        self.source_directory = Path("data/raw/vllm-0.10.1")
        self.output_directory = Path("data/processed")

    def chunk_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Split a Markdown file into chunks."""
        content = file_path.read_text(encoding="utf-8")

        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.05),
        )

        return self._create_chunks(
            file_path,
            content,
            splitter.split_text(content),
        )

    def chunk_python(self, file_path: Path) -> List[Dict[str, Any]]:
        """Split a Python file into chunks."""
        content = file_path.read_text(encoding="utf-8")

        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.05),
        )

        return self._create_chunks(
            file_path,
            content,
            splitter.split_text(content),
        )

    def chunk_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Split a plain-text file into chunks."""
        content = file_path.read_text(encoding="utf-8")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.05),
        )

        return self._create_chunks(
            file_path,
            content,
            splitter.split_text(content),
        )

    def _create_chunks(
        self,
        file_path: Path,
        content: str,
        chunks: List[str],
    ) -> List[Dict[str, Any]]:
        """Convert text chunks into indexed chunk records."""
        result = []
        search_start = 0

        for chunk in chunks:
            start_index = content.find(chunk, search_start)

            if start_index == -1:
                continue

            end_index = start_index + len(chunk)

            result.append(
                {
                    "file_path": str(file_path),
                    "first_character_index": start_index,
                    "last_character_index": end_index,
                    "content": chunk,
                }
            )

            search_start = start_index + 1

        return result

    def _find_files(self) -> List[Path]:
        """Find all supported source files."""
        supported_extensions = {".py", ".md", ".txt"}

        return [
            file_path
            for file_path in self.source_directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix in supported_extensions
        ]

    def _chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Choose the correct chunking strategy for a file."""
        if file_path.suffix == ".py":
            return self.chunk_python(file_path)

        if file_path.suffix == ".md":
            return self.chunk_markdown(file_path)

        if file_path.suffix == ".txt":
            return self.chunk_text(file_path)

        return []

    def _save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Save chunk metadata to disk."""
        self.output_directory.mkdir(parents=True, exist_ok=True)

        output_path = self.output_directory / "chunks.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4)

    def _build_bm25_index(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Build and save the BM25 index."""

        corpus = [chunk["content"] for chunk in chunks]
        
        retriever = bm25s.BM25(corpus=corpus)

        retriever.index(bm25s.tokenize(corpus))

        retriever.save("data/processed/bm25_index")


    def build_index(self) -> None:
        """Run the complete indexing pipeline."""
        all_chunks = []

        for file_path in self._find_files():
            try:
                chunks = self._chunk_file(file_path)
                all_chunks.extend(chunks)

            except (OSError, UnicodeDecodeError) as error:
                print(f"Skipping {file_path}: {error}")

        for chunk in all_chunks:
            assert len(chunk["content"]) <= self.max_chunk_size

        self._save_chunks(all_chunks)
        self._build_bm25_index(all_chunks)
