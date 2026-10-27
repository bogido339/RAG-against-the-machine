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

        self.iou_threshold: float = 0.05

        self.student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        self.dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )

    def _matches(self, retrieved, expected):
        if retrieved.file_path != expected.file_path:
            return False
        
        intersection = max(
            0,
            min(retrieved.last_character_index, expected.last_character_index)
            -
            max(retrieved.first_character_index, expected.first_character_index)
        )

        union = (
            retrieved.last_character_index
            - retrieved.first_character_index
            + expected.last_character_index
            - expected.first_character_index
            - intersection
        )
        
        return intersection / union >= self.iou_threshold

    def _question_recall(self, retrieved, expected):
        found = sum(
            any(self._matches(result, source) for result in retrieved)
            for source in expected
        )

        return found / len(expected)
    
    def evaluate(self) -> None:
        """Load validated data and calculate recall at several cutoffs."""

        results_by_id = {}
        for result in self.student_results.search_results:
            results_by_id.update({result.question_id: result})

        question_ids = set()
        for question in self.dataset.rag_questions:
            question_ids.add(question.question_id)

        scores = {}
        for k in (1, 3, 5, 10):
            total = 0.0
            for question in self.dataset.rag_questions:
                result = results_by_id.get(question.question_id)

                retrieved = result.retrieved_sources[:k] if result else []

                total += self._question_recall(retrieved, question.sources)
            
            scores.update({k: total / len(self.dataset.rag_questions)})
        
        for k, score in scores.items():
            print(f"Recall@{k}: {score:.3f} ({score:.1%})") 
