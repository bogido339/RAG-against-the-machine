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

        self.results_path = Path(student_search_results_path)
        self.data_path = Path(dataset_path)

    def evaluate(self) -> None:
        """Load validated data and calculate recall at several cutoffs."""

        with (
            self.results_path.open("r", encoding="utf-8") as file1,
            self.data_path.open("r", encoding="utf-8") as file2
        ):
            content1 = file1.read().strip()
            content2 = file2.read().strip()

            if not content1:
                raise EvaluatorError(
                    f"The file is empty"
                    f" (0 bytes): {self.results_path}")
            if not content2:
                raise EvaluatorError(
                    f"The file is empty"
                    f" (0 bytes): {self.data_path}")

            data1 = json.loads(content1)
            data2 = json.loads(content2)
        try:
            for item1, item2 in zip(data1["search_results"], data2["rag_questions"]):
                sources1 = item1["retrieved_sources"]
                sources2 = item2["sources"]
                data = [f["file_path"] for f in sources2]
                print("data", data)
                res = [x["file_path"] for x in sources1]
                print(res)
                print("---"*40)






        except KeyError as e:
            raise EvaluatorError(f"key not found: {e}")
