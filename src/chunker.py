from typing import List, Dict, Any
from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib



class BaseIndexer:
    """Abstract base class defining the indexing pipeline."""

    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Process and chunk markdown content based on structural boundaries."""

        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")

        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.05),
        )

        chunks = splitter.split_text(content)

        data = []
        search_start = 0

        for chunk in chunks:
            start_index = content.find(chunk, search_start)

            if start_index == -1:
                continue

            end_index = start_index + len(chunk)

            data.append({
                "file_path": str(file_path),
                "start_index": start_index,
                "end_index": end_index,
                "content": chunk
            })

            search_start = start_index + 1

        return data

    def chunk_python(self, file_path: str) -> List[Dict[str, Any]]:
        """Process and chunk Python source code."""

        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")

        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.05),
        )

        chunks = splitter.split_text(content)

        data = []
        search_start = 0

        for chunk in chunks:
            start_index = content.find(chunk, search_start)

            if start_index == -1:
                continue

            end_index = start_index + len(chunk)

            data.append({
                "file_path": str(file_path),
                "start_index": start_index,
                "end_index": end_index,
                "content": chunk
            })

            search_start = start_index + 1

        return data

    def build_index(self):
        """Traverse the corpus, chunk files, and save the complete index."""
        folder = Path("resources/vllm-0.10.1")

        all_chunks = []

        for file_path in folder.rglob("*"):

            if file_path.suffix == ".py":
                chunks = self.chunk_python(file_path)

            elif file_path.suffix == ".md":
                chunks = self.chunk_markdown(file_path)

            else:
                continue

            all_chunks.extend(chunks)
        
        for chunk in all_chunks:
            assert len(chunk["content"]) <= self.max_chunk_size

        Path("data").mkdir(exist_ok=True)

        with open("data/chunks.json", "w", encoding="utf-8") as file:
            json.dump(all_chunks, file, indent=4)

        return [chunk["content"] for chunk in all_chunks]


class Indexer():
    def __init__(self, chunks):
        self.chunks = chunks

    def build_TF_IDF_vector(self):

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(self.chunks)
        print(tfidf_matrix)

        joblib.dump(vectorizer, "data/tfidf_vectorizer.pkl")

        joblib.dump(tfidf_matrix, "data/tfidf_matrix.pkl")
