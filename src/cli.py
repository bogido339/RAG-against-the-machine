
import sys
from src.indexer import Indexer
from src.retriever import Retriever, DatasetRetriever
from src.generator import Generator, DatasetGenerator
from src.evaluator import Evaluator
from src.args_parser import Parser


class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def __init__(self, parser: Parser):
        self.parser = parser

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        self.parser.index_input(sys.argv[2:])

        indexer = Indexer(max_chunk_size)
        indexer.build_index()

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query."""
        self.parser.search_input(sys.argv[2:])

        retriever = Retriever()
        retriever.topk_search(query, k)

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 5
    ) -> None:
        """Run search over a whole dataset and save results."""
        self.parser.search_dataset_input(sys.argv[2:])

        dsr = DatasetRetriever()
        dsr.search_dataset(dataset_path, save_directory, k)

    def answer(self, query: str, k: int = 5) -> None:
        """Print an answer for one question."""
        self.parser.answer_input(sys.argv[2:])

        gen = Generator()
        gen.answer(query, k)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate and save answers from existing search results."""
        self.parser.answer_dataset_input(sys.argv[2:])

        datasetgen = DatasetGenerator()
        datasetgen.answer_dataset(
            student_search_results_path, save_directory
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print local recall scores against a ground-truth dataset."""
        self.parser.evaluate_input(sys.argv[2:])

        evaluator = Evaluator(
            student_search_results_path, dataset_path
        )
        evaluator.evaluate()
