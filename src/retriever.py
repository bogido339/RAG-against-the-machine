import json
from pathlib import Path
from typing import Any, Dict, List
import bm25s
from src.error_classes import RetrieverError, DatasetRetrieverError


class Retriever:
    """Load a BM25 index and retrieve source locations."""

    def __init__(
        self,
        chunks_path: str = "data/processed/chunks.json",
        bm25_index: str = "data/processed/bm25_index",
    ) -> None:
        """Load chunks and the saved index."""
        try:
            with open(chunks_path, "r", encoding="utf-8") as file:
                self.chunks = json.load(file)

            if not isinstance(self.chunks, list) or not self.chunks:
                raise RetrieverError("Chunks must be a non-empty JSON list.")

            self.chunks_by_text: Dict[str, Dict[str, Any]] = {}
            for chunk in self.chunks:
                self.chunks_by_text.setdefault(chunk["content"], chunk)

            self.retriever = bm25s.BM25.load(
                bm25_index,
                load_corpus=True,
            )
        except FileNotFoundError:
            raise RetrieverError(
                "Index files are missing. Run: "
                "uv run python -m src index --max_chunk_size 2000"
            )
        except PermissionError:
            raise RetrieverError("Permission denied while reading index files.")
        except (json.JSONDecodeError, UnicodeError):
            raise RetrieverError("Cannot read chunks JSON.")
        except (KeyError, TypeError):
            raise RetrieverError("Invalid chunk structure.")

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return at most top_k unique source locations."""
        if not query.strip():
            raise RetrieverError("Query must not be empty.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverError("top_k must be a positive integer.")

        results, _ = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=min(top_k, len(self.chunks)),
        )

        sources: List[Dict[str, Any]] = []
        seen = set()

        for result in results[0]:
            chunk = self.chunks_by_text.get(result["text"])
            if chunk is None:
                raise RetrieverError(
                    "Index and chunks do not match. Rebuild the index."
                )

            location = (
                chunk["file_path"],
                chunk["first_character_index"],
                chunk["last_character_index"],
                chunk["content"]
            )
            if location in seen:
                continue

            seen.add(location)
            sources.append({
                "file_path": location[0],
                "first_character_index": location[1],
                "last_character_index": location[2],
                "content": location[3]
            })

            if len(sources) >= top_k:
                break

        return sources

    def topk_search(self, query: str, top_k: int = 3) -> None:
        """Print retrieved source locations."""
        for source in self.search(query, top_k):
            print(
                f"{source['file_path']} "
                f"[{source['first_character_index']}:"
                f"{source['last_character_index']}]"
            )


class DatasetRetriever:
    """Search a dataset and save student search results."""

    def __init__(self) -> None:
        """Initialize the retriever."""
        self.retriever = Retriever()

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        top_k: int = 5,
    ) -> None:
        """Save results inside the requested output directory."""
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverError("top_k must be a positive integer.")

        try:
            with open(dataset_path, "r", encoding="utf-8") as file:
                dataset = json.load(file)

            results = []
            for query in dataset["rag_questions"]:
                results.append({
                    "question_id": query["question_id"],
                    "question": query["question"],
                    "retrieved_sources": self.retriever.search(
                        query["question"], top_k
                    ),
                })
        except FileNotFoundError:
            raise DatasetRetrieverError(f"Dataset does not exist: {dataset_path}")
        except PermissionError:
            raise DatasetRetrieverError(f"Cannot read dataset: {dataset_path}")
        except (json.JSONDecodeError, UnicodeError):
            raise DatasetRetrieverError("Cannot read dataset JSON.")
        except (KeyError, TypeError):
            raise DatasetRetrieverError("Invalid dataset structure.")

        output_directory = Path(save_directory)
        output_path = output_directory / Path(dataset_path).name

        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {"search_results": results, "k": top_k},
                    file,
                    indent=4,
                    ensure_ascii=False,
                )
        except OSError as error:
            raise DatasetRetrieverError(
                f"Cannot save results to {output_path}: {error}"
            ) from error

        print(f"Saved student_search_results to {output_path}")
