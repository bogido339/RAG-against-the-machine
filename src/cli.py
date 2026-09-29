from src.indexer import Indexer
from src.retriever import Retriever, DatasetRetriever
from src.generator import Generator, DatasetGenerator

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


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
        retriver.topk_search(query, k)

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
