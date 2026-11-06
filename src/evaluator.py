from pathlib import Path
from pydantic import ValidationError
from src.error_classes import EvaluatorError
from src.models import RagDataset, StudentSearchResults

class Evaluator:
    """Measure retrieval recall against ground-truth sources."""

    def __init__(self, student_search_results_path: str, dataset_path: str) -> None:
        """Initialize evaluator and safely load JSON data."""
        self.iou_threshold: float = 0.05

        try:
            self.student_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text(encoding="utf-8")
            )
            self.dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text(encoding="utf-8")
            )
        except OSError as e:
            raise EvaluatorError(f"File system error for '{e.filename}': {e.strerror}")
        except ValidationError:
            raise EvaluatorError(
                "Invalid dataset structure: "
                "The JSON file does not match the required format."
            )
        except Exception as e:
            raise EvaluatorError(f"Failed to load evaluation data: {e}")

    def _matches(self, retrieved, expected) -> bool:
        if retrieved.file_path != expected.file_path:
            return False
        
        intersection = max(
            0,
            min(retrieved.last_character_index, expected.last_character_index)
            - max(retrieved.first_character_index, expected.first_character_index)
        )

        union = (
            (retrieved.last_character_index - retrieved.first_character_index)
            + (expected.last_character_index - expected.first_character_index)
            - intersection
        )
        
        if union <= 0:
            return False
            
        return intersection / union >= self.iou_threshold

    def _question_recall(self, retrieved, expected) -> float:
        if not expected:
            return 0.0

        found = sum(
            any(self._matches(result, source) for result in retrieved)
            for source in expected
        )

        return found / len(expected)
    
    def evaluate(self) -> None:
        """Load validated data and calculate recall at several cutoffs."""
        
        total_questions = len(self.dataset.rag_questions)
        if total_questions == 0:
            print("Warning: The dataset is empty. Cannot calculate recall.")
            return

        results_by_id = {
            result.question_id: result 
            for result in self.student_results.search_results
        }

        scores = {}
        for k in (1, 3, 5, 10):
            total_recall = 0.0
            
            for question in self.dataset.rag_questions:
                result = results_by_id.get(question.question_id)
                retrieved = result.retrieved_sources[:k] if result else []
                
                expected_sources = getattr(question, 'sources', [])
                
                total_recall += self._question_recall(retrieved, expected_sources)
            
            scores[k] = total_recall / total_questions
        
        for k, score in scores.items():
            print(f"Recall@{k}: {score:.3f} ({score:.1%})")
