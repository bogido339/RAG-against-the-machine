import json
from pathlib import Path
from typing import Any, Dict, List
import bm25s


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
                raise ValueError("Chunks must be a non-empty JSON list.")

            self.chunks_by_text: Dict[str, Dict[str, Any]] = {}
            for chunk in self.chunks:
                self.chunks_by_text.setdefault(chunk["content"], chunk)

            self.retriever = bm25s.BM25.load(
                bm25_index,
                load_corpus=True,
            )
        except FileNotFoundError as error:
            raise ValueError(
                "Index files are missing. Run: "
                "uv run python -m src index --max_chunk_size 2000"
            ) from error
        except PermissionError as error:
            raise ValueError(
                "Permission denied while reading index files."
            ) from error
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Cannot read chunks JSON.") from error
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid chunk structure.") from error

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return at most top_k unique source locations."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("k must be greater than zero.")

        results, _ = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=min(top_k, len(self.chunks)),
        )

        sources: List[Dict[str, Any]] = []
        seen = set()

        for result in results[0]:
            chunk = self.chunks_by_text.get(result["text"])
            if chunk is None:
                raise ValueError(
                    "Index and chunks do not match. Rebuild the index."
                )

            location = (
                chunk["file_path"],
                chunk["first_character_index"],
                chunk["last_character_index"],
            )
            if location in seen:
                continue

            seen.add(location)
            sources.append({
                "file_path": location[0],
                "first_character_index": location[1],
                "last_character_index": location[2],
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
        k: int = 5,
    ) -> None:
        """Save results inside the requested output directory."""
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        try:
            with open(dataset_path, "r", encoding="utf-8") as file:
                dataset = json.load(file)

            results = []
            for query in dataset["rag_questions"]:
                results.append({
                    "question_id": query["question_id"],
                    "question": query["question"],
                    "retrieved_sources": self.retriever.search(
                        query["question"], k
                    ),
                })
        except FileNotFoundError as error:
            raise ValueError(
                f"Dataset does not exist: {dataset_path}"
            ) from error
        except PermissionError as error:
            raise ValueError(
                f"Cannot read dataset: {dataset_path}"
            ) from error
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Cannot read dataset JSON.") from error
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid dataset structure.") from error

        output_directory = Path(save_directory)
        output_path = output_directory / Path(dataset_path).name

        try:
            output_directory.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {"search_results": results, "k": k},
                    file,
                    indent=4,
                    ensure_ascii=False,
                )
        except OSError as error:
            raise ValueError(
                f"Cannot save results to {output_path}: {error}"
            ) from error

        print(f"Saved student_search_results to {output_path}")
