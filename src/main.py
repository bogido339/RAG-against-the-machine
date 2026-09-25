import sys
import fire
from typing import Optional

from src.chunker import BaseIndexer, Indexer
from src.retriever import Retriever
from src.generator import Generator


class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/.
        """
        Indexer(BaseIndexer(max_chunk_size).build_index()).build_TF_IDF_vector()

    def search(self, query: str, k: int = 5) -> None:
        """
        Return the top-k sources for a single query.
        """
        retriver = Retriever()
        top_k = retriver.search(query, k)
        print(top_k)

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5) -> None:
        """
        Run search over a whole dataset and write a StudentSearchResults JSON file.
        """
        pass

    def answer(self, query: str, k: int = 5) -> None:
        """
        Answer a single query using the retrieved context.
        """
        generator = Generator()
        generator.answer(query, k)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """
        Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file.
        """
        pass

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Report your own recall@k against a ground-truth dataset for local testing.
        """
        pass
    def jj(self, number):
        print("lsebar")


def main() -> None:
    """Main entry point for the CLI."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit()


if __name__ == '__main__':
    main()
