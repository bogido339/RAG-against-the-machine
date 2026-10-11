from pathlib import Path

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluator:
    """Measure retrieval recall against ground-truth sources."""

    def __init__(self, iou_threshold: float = 0.05) -> None:
        """Set the minimum overlap required for a source match."""
        self.iou_threshold = iou_threshold

    def evaluate(
        self,student_search_results_path: str,dataset_path: str
    ) -> None:
        """Load validated data and calculate recall at several cutoffs."""

        pass