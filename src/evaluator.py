from pathlib import Path
from src.error_classes import EvaluatorError
import json

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluator:
    """Measure retrieval recall against ground-truth sources."""

    def __init__(self, student_search_results_path: str, dataset_path: str) -> None:
        """Set the minimum overlap required for a source match."""

        self.student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        self.dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )

    def evaluate(self) -> None:
        """Load validated data and calculate recall at several cutoffs."""

        pass


