from typing import List, Dict, Any
from pathlib import Path
import json
import stat
import sys

import bm25s
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

from src.error_classes import IndexerError


class Indexer:
    """Build the searchable index from the source codebase."""

    def __init__(self, max_chunk_size: int = 2000):
        if type(max_chunk_size) is not int or not 1 <= max_chunk_size <= 2000:
            raise IndexerError(
                "max_chunk_size must be an integer between 1 and 2000."
            )

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
                raise IndexerError(
                    f"Could not locate a generated chunk in source file: "
                    f"{file_path}"
                )

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

        return sorted(
            file_path
            for file_path in self.source_directory.rglob("*")
            if file_path.is_file()
            and file_path.suffix in supported_extensions
        )

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
        output_path = self.output_directory / "chunks.json"

        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(chunks, file, indent=4)

        except OSError as error:
            raise IndexerError(
                f"Could not save chunks to '{output_path}': {error}"
            ) from error

    def _build_bm25_index(
        self,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Build and save the BM25 index."""
        index_path = self.output_directory / "bm25_index"

        if not chunks:
            raise IndexerError(
                "Cannot build the BM25 index: no chunks were provided."
            )

        corpus = [chunk["content"] for chunk in chunks]

        try:
            tokens = bm25s.tokenize(corpus)

            if not any(tokens.ids):
                raise IndexerError(
                    "Cannot build the BM25 index: "
                    "no searchable tokens were found."
                )

            retriever = bm25s.BM25(corpus=corpus)
            retriever.index(tokens)

        except MemoryError as error:
            raise IndexerError(
                "Not enough memory to build the BM25 index."
            ) from error

        except (ValueError, RuntimeError) as error:
            raise IndexerError(
                f"Could not build the BM25 index: {error}"
            ) from error

        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            retriever.save(str(index_path))

        except OSError as error:
            raise IndexerError(
                f"Could not save the BM25 index to '{index_path}': {error}"
            ) from error

    def build_index(self) -> None:
        """Run the complete indexing pipeline."""
        try:
            source_info = self.source_directory.stat()

        except FileNotFoundError as error:
            raise IndexerError(
                f"Source directory not found: {self.source_directory}"
            ) from error

        except OSError as error:
            raise IndexerError(
                f"Could not access source directory "
                f"'{self.source_directory}': {error}"
            ) from error

        if not stat.S_ISDIR(source_info.st_mode):
            raise IndexerError(
                f"Expected a directory: {self.source_directory}"
            )

        try:
            files = self._find_files()

        except OSError as error:
            raise IndexerError(
                f"Could not scan source directory "
                f"'{self.source_directory}': {error}"
            ) from error

        if not files:
            raise IndexerError(
                f"No supported files (.py, .md, .txt) were found in: "
                f"{self.source_directory}"
            )

        all_chunks = []

        for file_path in files:
            try:
                chunks = self._chunk_file(file_path)
                all_chunks.extend(chunks)

            except UnicodeDecodeError as error:
                print(
                    f"Skipping '{file_path}': invalid UTF-8 content. {error}",
                    file=sys.stderr,
                )

            except OSError as error:
                print(
                    f"Skipping '{file_path}': could not read the file. {error}",
                    file=sys.stderr,
                )

        if not all_chunks:
            raise IndexerError(
                "Cannot build the index: no usable chunks were produced. "
                "Source files may be empty, unreadable, "
                "or contain only whitespace."
            )

        self._save_chunks(all_chunks)
        self._build_bm25_index(all_chunks)