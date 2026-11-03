import json
from pathlib import Path
from typing import Any, List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.retriever import Retriever
from src.error_classes import GeneratorErrro, DatasetGeneratorError
from tqdm import tqdm
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer, MinimalSource


class Qwen:
    """Handle text generation using Qwen."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, user_input: str) -> str:
        """Generate a response from Qwen."""
        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt")

        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024
        )[0][len(inputs.input_ids[0]):]

        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        )


class Generator:
    """Generate answers using retrieved context."""

    def __init__(self):
        self.model = Qwen()
        self.retriever = Retriever()

    def _load_chunks(self, sources: List[MinimalSource]) -> List[str]:
        chunks = []

        for source in sources:
            file_path = Path(source.file_path)

            content = file_path.read_text(encoding="utf-8")
    
            start = source.first_character_index
            end = source.last_character_index

            chunks.append(content[start:end])

        return chunks

    def get_prompt(self, query: str, top_chunks: list[str]) -> str:
        """Build the RAG prompt."""
        context = "\n\n".join(top_chunks)

        return f"""Answer the question based only on this context.

Context:
{context}

Question:
{query}

Answer:"""

    def generate(self, query: str, chunks: List[MinimalSource]) -> str:

        prompt = self.get_prompt(query, self._load_chunks(chunks))

        return self.model.generate_response(prompt)

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve relevant chunks and answer one question."""

        top_chunks = self.retriever.search(query, k)

        print(self.generate(query, top_chunks))


class DatasetGenerator:
    """Generate answers for a search-results dataset."""

    def __init__(self):
        self.generator = Generator()

    def _load_search_results(
            self, input_path: Path
    ) -> StudentSearchResults:
        
        if not input_path.exists():
            raise DatasetGeneratorError(
                f"The search results file does not exist: {input_path}")
            
        try:
            with input_path.open("r", encoding="utf-8") as file:
                content = file.read().strip()
                if not content:
                    raise DatasetGeneratorError(
                        f"The search results file is empty"
                        f" (0 bytes): {input_path}")

                #------------------------------------------------------
                data = StudentSearchResults.model_validate_json(content)

        except PermissionError as e:
            raise DatasetGeneratorError(
                f"Permission denied: You do not have read access to "
                f"'{input_path}'. Check file permissions."
            )
        except json.decoder.JSONDecodeError as e:
            raise DatasetGeneratorError(
                f"Failed to parse JSON from '{input_path}'. "
                f"The file contains invalid JSON. Details: {e}"
            )

        return data
    
    def _process_dataset(
            self, data: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:

        results = []

        try:
            for query in tqdm(data.search_results[:2]):
                try:
                    question_id = query.question_id
                    question = query.question
                    retrieved_sources = query.retrieved_sources

                    answer = self.generator.generate(question, retrieved_sources)

                    results.append(
                        {
                            "question_id": question_id,
                            "question": question,
                            "retrieved_sources": retrieved_sources,
                            "answer": answer
                        }
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to process question ID "
                        f"'{query.question_id}'. Error: {e}")

            output = StudentSearchResultsAndAnswer(
                search_results=results,
                k=data.k
            )

            return output

        except Exception as e:
            raise DatasetGeneratorError(
                f"Critical error while processing dataset: {e}")

    def _save(self, data: StudentSearchResultsAndAnswer,
        input_path: Path, save_directory: Path
    ) -> None:
        """Save generated dataset to a JSON file."""
        output_path = save_directory / input_path.name

        try:
            save_directory.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(
                    data.model_dump(mode="json"),
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            print(f"Saved results to {output_path}")

        except OSError as e:
            raise DatasetGeneratorError(
                f"Failed to save dataset to '{output_path}': {e}"
            ) from e

    def _validate_input(
            self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Validate dataset input and output paths."""

        if not isinstance(student_search_results_path, str):
            raise DatasetGeneratorError(
                "student_search_results_path must be a string."
            )

        if not student_search_results_path.strip():
            raise DatasetGeneratorError(
                "student_search_results_path must not be empty."
            )

        if not isinstance(save_directory, str):
            raise DatasetGeneratorError(
                "save_directory must be a string."
            )

        if not save_directory.strip():
            raise DatasetGeneratorError(
                "save_directory must not be empty."
            )

        input_path = Path(student_search_results_path)
        output_dir = Path(save_directory)

        if not input_path.is_file():
            raise DatasetGeneratorError(
                f"Dataset file does not exist: '{input_path}'"
            )

        if input_path.suffix.lower() != ".json":
            raise DatasetGeneratorError(
                "Dataset file must have a .json extension."
            )

        if output_dir.exists() and not output_dir.is_dir():
            raise DatasetGeneratorError(
                f"Output path is not a directory: '{output_dir}'"
            )

        if not output_dir.is_absolute() and not output_dir.parts:
            raise DatasetGeneratorError(
                "Invalid output directory."
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        """Generate answers for a dataset and save the results."""

        self._validate_input(
            student_search_results_path,
            save_directory
        )

        input_path = Path(student_search_results_path)
        output_dir = Path(save_directory)

        try:
            data = self._load_search_results(input_path)

            output = self._process_dataset(data)

            self._save(output, input_path, output_dir)

        except DatasetGeneratorError:
            raise

        except (OSError, ValueError, TypeError) as error:
            raise DatasetGeneratorError(
                f"Failed to process dataset '{input_path}': {error}"
            )
