import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.retriever import Retriever
from src.error_classes import GeneratorError, DatasetGeneratorError
from tqdm import tqdm
from src.models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSource,
    MinimalAnswer
)
from pydantic import ValidationError
from typing import cast


class Qwen:
    """Handle text generation using Qwen."""

    def __init__(
            self, model_name: str = "Qwen/Qwen3-0.6B",
            max_new_tokens: int = 1024
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.max_context_tokens = self.model.config.max_position_embeddings
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = self.max_context_tokens - self.max_new_tokens

    def count_tokens(self, text: str) -> int:
        """Utility to count tokens for a given text."""
        return len(self.tokenizer.encode(text))

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

        input_length = len(inputs.input_ids[0])

        if input_length > self.max_prompt_tokens:
            raise GeneratorError(
                f"Context window exceeded! "
                f"The prompt has {input_length} tokens, "
                f"but the maximum allowed is {self.max_context_tokens}."
            )

        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024
        )[0][input_length:]

        return cast(
            str,
            self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            )
        )


class Generator:
    """Generate answers using retrieved context."""

    def __init__(self) -> None:
        try:
            self.model = Qwen()
            self.retriever = Retriever()
        except Exception as e:
            raise GeneratorError(
                f"Error initializing Generator components: {e}")

    def _load_chunks(self, sources: list[MinimalSource]) -> list[str]:
        chunks = []

        for source in sources:
            file_path = Path(source.file_path)

            try:
                content = file_path.read_text(encoding="utf-8")
                start = source.first_character_index
                end = source.last_character_index
                chunk_text = content[start:end].strip()

                if not chunk_text:
                    print(
                        f"Warning: Extracted chunk from "
                        f"{file_path} is empty. Skipping..."
                    )
                    continue

                chunks.append(chunk_text)

            except Exception as e:
                print(f"Warning: Failed to read {file_path}. Error: {e}")
                continue

        return chunks

    def get_prompt(self, search_result: str, top_chunks: list[str]) -> str:
        """Build the RAG prompt."""
        context = "\n\n".join(top_chunks)

        return f"""Answer the question based only on this context.

Context:
{context}

Question:
{search_result}

Answer:"""

    def generate(self, query: str, chunks: list[MinimalSource]) -> str:

        loaded_chunks = self._load_chunks(chunks)

        if not loaded_chunks:
            raise GeneratorError(
                "No valid content could be extracted "
                "from the retrieved sources. "
                "Files might be missing or empty."
            )
        prompt = self.get_prompt(query, loaded_chunks)

        while self.model.count_tokens(prompt) > self.model.max_prompt_tokens:
            if not loaded_chunks:
                raise GeneratorError(
                    "The query itself exceeds the maximum context window."
                )

            print(
                "Warning: Context window exceeded. "
                "Dropping the lowest ranked chunk..."
            )
            loaded_chunks.pop()

            prompt = self.get_prompt(query, loaded_chunks)

        return self.model.generate_response(prompt)

    def answer(self, search_result: str, k: int = 5) -> None:
        """Retrieve relevant chunks and answer one question."""

        top_chunks = self.retriever.search(search_result, k)

        print(self.generate(search_result, top_chunks))


class DatasetGenerator:
    """Generate answers for a search-results dataset."""

    def __init__(self) -> None:
        self.generator: Generator = Generator()

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

                data = StudentSearchResults.model_validate_json(content)

        except PermissionError:
            raise DatasetGeneratorError(
                f"Permission denied: You do not have read access to "
                f"'{input_path}'. Check file permissions."
            )
        except (json.decoder.JSONDecodeError, ValidationError) as e:
            raise DatasetGeneratorError(f"Invalid dataset format: {e}")

        return data

    def _process_dataset(
            self, data: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:

        results: list[MinimalAnswer] = []

        try:
            for search_result in tqdm(data.search_results):
                try:
                    question_id = search_result.question_id
                    question = search_result.question
                    retrieved_sources = search_result.retrieved_sources

                    answer = self.generator.generate(
                        question, retrieved_sources
                    )

                    results.append(
                        MinimalAnswer(
                            question_id=question_id,
                            question=question,
                            retrieved_sources=retrieved_sources,
                            answer=answer
                        )
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to process question ID "
                        f"'{question_id}'. Error: {e}")

            output = StudentSearchResultsAndAnswer(
                search_results=results,
                k=data.k
            )

            return output

        except Exception as e:
            raise DatasetGeneratorError(
                f"Critical error while processing dataset: {e}")

    def _save(
        self,
        data: StudentSearchResultsAndAnswer,
        input_path: Path,
        save_directory: Path
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
                f"Failed to save dataset to '{output_path}': {e}")

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
            self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for a dataset and save the results."""

        self._validate_input(student_search_results_path, save_directory)

        input_path = Path(student_search_results_path)
        output_dir = Path(save_directory)

        try:
            data = self._load_search_results(input_path)

            output = self._process_dataset(data)

            self._save(output, input_path, output_dir)

        except DatasetGeneratorError:
            raise

        except Exception as error:
            raise DatasetGeneratorError(
                f"Failed to process dataset '{input_path}': {error}"
            )
