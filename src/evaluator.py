
from pathlib import Path

from pydantic import ValidationError

from src.error_classes import EvaluatorError
from src.models import (
    RagDataset,
    StudentSearchResults,
    MinimalSource,
    AnsweredQuestion,
)


class Evaluator:
    """Measure retrieval recall against ground-truth sources"""

    def __init__(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Initialize the evaluator and load validated JSON data"""

        self.iou_threshold: float = 0.05

        try:
            self.student_results = (
                StudentSearchResults.model_validate_json(
                    Path(student_search_results_path).read_text(
                        encoding="utf-8"
                    )
                )
            )

            self.dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )

        except OSError as e:
            raise EvaluatorError(
                f"File system error: {e}"
            )

        except ValidationError as e:
            raise EvaluatorError(
                f"Invalid dataset structure: {e}"
            )

        except Exception as e:
            raise EvaluatorError(
                f"Failed to load evaluation data: {e}"
            )

    def _matches(
        self, retrieved: MinimalSource, expected: MinimalSource
    ) -> bool:
        """Check whether two sources match using file path and IoU"""

        if retrieved.file_path != expected.file_path:
            return False

        retrieved_start = retrieved.first_character_index
        retrieved_end = retrieved.last_character_index

        expected_start = expected.first_character_index
        expected_end = expected.last_character_index

        if retrieved_start < 0 or expected_start < 0:
            return False

        if retrieved_end <= retrieved_start:
            return False

        if expected_end <= expected_start:
            return False

        intersection = max(
            0,
            min(retrieved_end, expected_end)
            - max(retrieved_start, expected_start)
        )

        union = (
            (retrieved_end - retrieved_start)
            + (expected_end - expected_start)
            - intersection
        )

        if union <= 0:
            return False

        return intersection / union >= self.iou_threshold

    def _question_recall(
        self,
        retrieved: list[MinimalSource],
        expected: list[MinimalSource]
    ) -> float:
        """Calculate the fraction of expected sources matched by retrieval"""

        if not expected:
            return 0.0

        found = sum(
            any(
                self._matches(result, source)
                for result in retrieved
            )
            for source in expected
        )

        return found / len(expected)

    def _validate_dataset(self) -> None:
        """Validate that all evaluation questions have ground-truth data"""

        if not self.dataset.rag_questions:
            raise EvaluatorError(
                "The evaluation dataset is empty"
            )

        question_ids: set[str] = set()

        for question in self.dataset.rag_questions:
            if not isinstance(question, AnsweredQuestion):
                raise EvaluatorError(
                    f"Question '{question.question_id}' "
                    "has no ground-truth sources"
                )

            if question.question_id in question_ids:
                raise EvaluatorError(
                    f"Duplicate question ID: '{question.question_id}'"
                )

            question_ids.add(question.question_id)

    def evaluate(self) -> None:
        """Calculate and display average retrieval
            recall at multiple cutoffs"""

        self._validate_dataset()

        total_questions = len(self.dataset.rag_questions)

        results_by_id = {}

        for search_result in self.student_results.search_results:
            if search_result.question_id in results_by_id:
                raise EvaluatorError(
                    f"Duplicate search result ID: "
                    f"'{search_result.question_id}'."
                )

            results_by_id[search_result.question_id] = search_result

        scores: dict[int, float] = {}

        for k in (1, 3, 5, 10):
            total_recall = 0.0

            for question in self.dataset.rag_questions:
                result = results_by_id.get(question.question_id)

                retrieved = (
                    result.retrieved_sources[:k]
                    if result is not None
                    else []
                )

                if not isinstance(question, AnsweredQuestion):
                    raise EvaluatorError(
                        f"Missing ground-truth sources for "
                        f"question '{question.question_id}'"
                    )

                total_recall += self._question_recall(
                    retrieved,
                    question.sources
                )

            scores[k] = total_recall / total_questions

        for k, score in scores.items():
            print(f"Recall@{k}: {score:.3f} ({score:.1%})")
