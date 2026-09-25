import json
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


def chunk_markdown(self, file_path: str) -> None:
    """Chunk a Markdown file and store each chunk's character positions."""

    path = Path(file_path)
    file_content = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=self.max_chunk_size,
        chunk_overlap=int(self.max_chunk_size * 0.05),
    )

    chunks = splitter.split_text(file_content)

    chunks_data: list[dict[str, Any]] = []
    search_start = 0

    for chunk in chunks:
        start_index = file_content.find(chunk, search_start)

        if start_index == -1:
            continue

        end_index = start_index + len(chunk)

        chunks_data.append(
            {
                "file_path": str(path),
                "start_index": start_index,
                "end_index": end_index,
                "text": chunk,
            }
        )

        search_start = start_index + 1

    output_path = Path("data/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(chunks_data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    