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
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Load validated data and calculate recall at several cutoffs."""
        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )

        if student_results.k <= 0:
            raise ValueError("k must be greater than zero.")

        results_by_id = {}
        for result in student_results.search_results:
            if result.question_id in results_by_id:
                raise ValueError(
                    f"Duplicate question ID: {result.question_id}"
                )
            if len(result.retrieved_sources) > student_results.k:
                raise ValueError(
                    f"Question {result.question_id} has too many sources."
                )

            for source in result.retrieved_sources:
                self._validate_source(source)
                if (
                    source.last_character_index
                    - source.first_character_index
                ) > 2000:
                    raise ValueError("Retrieved source exceeds 2000 characters.")

            results_by_id[result.question_id] = result

        questions = [
            question
            for question in dataset.rag_questions
            if isinstance(question, AnsweredQuestion) and question.sources
        ]
        if not questions:
            raise ValueError("Dataset has no ground-truth sources.")

        question_ids = set()
        for question in questions:
            if question.question_id in question_ids:
                raise ValueError(
                    f"Duplicate reference question ID: {question.question_id}"
                )
            question_ids.add(question.question_id)

            for source in question.sources:
                self._validate_source(source)

        cutoffs = sorted({
            k for k in (1, 3, 5, 10, student_results.k)
            if k <= student_results.k
        })
        scores = {}

        for k in cutoffs:
            total = 0.0
            for question in questions:
                result = results_by_id.get(question.question_id)
                retrieved = (
                    result.retrieved_sources[:k] if result else []
                )
                total += self._question_recall(
                    retrieved, question.sources
                )
            scores[k] = total / len(questions)

        for k, score in scores.items():
            print(f"Recall@{k}: {score:.3f} ({score:.1%})")

    def _validate_source(self, source: MinimalSource) -> None:
        """Reject invalid source locations."""
        if (
            not source.file_path
            or source.first_character_index < 0
            or source.last_character_index <= source.first_character_index
        ):
            raise ValueError("Invalid source path or character range.")

    def _question_recall(
        self,
        retrieved: list[MinimalSource],
        expected: list[MinimalSource],
    ) -> float:
        """Count each reference source at most once."""
        found = sum(
            any(self._matches(result, source) for result in retrieved)
            for source in expected
        )
        return found / len(expected)

    def _matches(
        self,
        retrieved: MinimalSource,
        expected: MinimalSource,
    ) -> bool:
        """Check the exact file path and interval IoU."""
        if retrieved.file_path != expected.file_path:
            return False

        intersection = max(
            0,
            min(
                retrieved.last_character_index,
                expected.last_character_index,
            )
            - max(
                retrieved.first_character_index,
                expected.first_character_index,
            ),
        )
        union = (
            retrieved.last_character_index
            - retrieved.first_character_index
            + expected.last_character_index
            - expected.first_character_index
            - intersection
        )

        return intersection / union >= self.iou_threshold
