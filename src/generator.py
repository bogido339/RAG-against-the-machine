# from transformers import AutoModelForCausalLM, AutoTokenizer
# from src.retriever import Retriever


# class QwenRAG:
#     def __init__(self, model_name="Qwen/Qwen3-0.6B"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)

#     def generate_response(self, user_input):
#         messages = [{"role": "user", "content": user_input}]

#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         inputs = self.tokenizer(text, return_tensors="pt")
#         response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
#         response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
#         return response


# class Generator:
#     def __init__(self):
#         self.model = QwenRAG()

#     def get_prompt(self, query, top_chunks):

#         return (f"""Answer the question based only on this context:

# Context:
# {chr(10).join(top_chunks)}

# Question: {query}

# Answer:""")


#     def answer(self, query, k):
#         retriver = Retriever()
#         top_k = retriver.search(query, k)
        
#         prompt = self.get_prompt(query, [chunk["content"] for chunk in top_k])

#         respons = self.model.generate_response(prompt)
#         print(respons)
#         print("##" * 40)
#         for k in top_k:
#             print(k)
#             print("--" * 40)
#             print("")
    
# class DatasetGenerator():
#     def __init__(self):
#         self.generator = Generator()

#     def answer_dataset(self, student_search_results_path, save_directory):
#         with open("student_search_results_path", "r") as file:
#             for query in file:
#                 question_id = query["question_id"]
#                 question = query["question"]

#                 answer = self.generator.answer(question, 5)

import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retriever import Retriever


class QwenRAG:
    """Handle text generation using Qwen."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, user_input: str) -> str:
        """Generate a response from Qwen."""
        messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )[0][len(inputs.input_ids[0]):]

        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )


class Generator:
    """Generate answers using retrieved context."""

    def __init__(self):
        self.model = QwenRAG()

    def get_prompt(
        self,
        query: str,
        top_chunks: list[str],
    ) -> str:
        """Build the RAG prompt."""
        context = "\n\n".join(top_chunks)

        return f"""Answer the question based only on this context.

Context:
{context}

Question:
{query}

Answer:"""

    def generate(
        self,
        query: str,
        chunks: list[str],
    ) -> str:
        """Generate an answer from already retrieved chunks."""
        prompt = self.get_prompt(query, chunks)

        return self.model.generate_response(prompt)

    def answer(
        self,
        query: str,
        k: int = 5,
    ) -> str:
        """Retrieve relevant chunks and answer one question."""
        retriever = Retriever()

        top_chunks = retriever.search(query, k)

        contents = [
            chunk["content"]
            for chunk in top_chunks
        ]

        return self.generate(query, contents)


class DatasetGenerator:
    """Generate answers for a search-results dataset."""

    def __init__(self):
        self.generator = Generator()

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers from existing search results."""

        input_path = Path(student_search_results_path)

        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        results = []

        for query in data["search_results"]:
            question_id = query["question_id"]
            question = query["question"]
            retrieved_sources = query["retrieved_sources"]

            chunks = self._load_chunks(retrieved_sources)

            answer = self.generator.generate(
                question,
                chunks,
            )

            results.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "retrieved_sources": retrieved_sources,
                    "answer": answer,
                }
            )

        output = {
            "search_results": results,
            "k": data["k"],
        }

        self._save(
            output,
            input_path,
            Path(save_directory),
        )

    def _load_chunks(
        self,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        """Load source text using its character positions."""
        chunks = []

        for source in sources:
            file_path = Path(source["file_path"])

            content = file_path.read_text(encoding="utf-8")

            start = source["first_character_index"]
            end = source["last_character_index"]

            chunks.append(content[start:end])

        return chunks

    def _save(
        self,
        data: dict[str, Any],
        input_path: Path,
        save_directory: Path,
    ) -> None:
        """Save generated dataset using the original filename."""
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = save_directory / input_path.name

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"Saved results to {output_path}")