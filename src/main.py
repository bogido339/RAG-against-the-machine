import sys
import fire
from typing import Optional

from src.indexer import Indexer
from src.retriever import Retriever, DatasetRetriever
from src.generator import Generator, DatasetGenerator


class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        indexer = Indexer(max_chunk_size)
        indexer.build_index()

    def search(self, query: str, k: int = 5) -> None:
        """
        Return the top-k sources for a single query.
        """
        retriver = Retriever()
        top_k_chunks = retriver.search(query, k)
        for chunk in top_k_chunks:
            print(chunk["file_path"], end=" ")
            print(f"[{chunk["first_character_index"]}", end=":")
            print(chunk["last_character_index"], end="]\n")

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        """
        Run search over a whole dataset and write a StudentSearchResults JSON file.
        """
        dsr = DatasetRetriever()
        dsr.search_dataset(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5) -> None:
        """
        Answer a single query using the retrieved context.
        """
        gen = Generator()
        gen.answer(query, k)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.
        """
        gen = DatasetGenerator()
        gen.answer_dataset(student_search_results_path, save_directory)

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth dataset for local testing.
        """
        pass


def main() -> None:
    """Main entry point for the CLI."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
