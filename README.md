*This project has been created as part of the 42 curriculum by mbougajd.*
 
# RAG against the machine
 
## Description
 
This project implements a **Retrieval-Augmented Generation (RAG)** system that answers
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase without
retraining or fine-tuning a model.
 
Instead of relying on what a language model already "knows," the system:
 
1. **Indexes** the vLLM source tree (Python code + Markdown docs) into searchable chunks.
2. **Retrieves** the most relevant chunks for a given question using a lexical search
   method (BM25 / TF-IDF).
3. **Augments** the model's context window with those retrieved chunks.
4. **Generates** a grounded, natural-language answer using `Qwen/Qwen3-0.6B`.
Retrieval quality is measured with **recall@k**, and every stage of the pipeline is
exposed through a CLI so it can be run and evaluated end-to-end.
 
## Instructions
 
### Prerequisites
 
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) as the package/project manager
### Installation
 
```bash
git clone <your-repo-url>
cd rag-against-the-machine
make install
```
 
`make install` runs `uv sync`, which creates the virtual environment and installs all
dependencies from `pyproject.toml` / `uv.lock`.
 
### Running the pipeline
 
All commands go through the CLI, exposed as `uv run python -m src <command>`.
 
| Makefile rule | What it does |
|---|---|
| `make install` | Installs dependencies via `uv sync` |
| `make run` | Runs the main entry point |
| `make debug` | Runs the main entry point under `pdb` |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, and other build artifacts |
| `make lint` | Runs `flake8 .` and `mypy .` with the required flags |
| `make lint-strict` | Runs `flake8 .` and `mypy . --strict` |
 
Full pipeline, step by step:
 
```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000
 
# 2. Search a single query
uv run python -m src search "How to configure the OpenAI server?" --k 5
 
# 3. Search a full dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
 
# 4. Generate an answer for a single query
uv run python -m src answer "How to configure the OpenAI server?" --k 5
 
# 5. Generate answers for a full dataset
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
 
# 6. Evaluate against a ground-truth dataset (local iteration only)
uv run python -m src evaluate \
  --student_search_results_path <path> \
  --dataset_path <path>
```
 
## System Architecture
 
```
data/raw/ (vLLM repo)
     │
     ▼
 ┌─────────┐    chunking    ┌──────────┐    BM25/TF-IDF   ┌───────────┐
 │ Indexer │ ─────────────▶ │  Index   │ ───────────────▶ │ Retriever │
 └─────────┘                └──────────┘                  └───────────┘
                                                                  │ top-k sources
                                                                  ▼
                                                          ┌───────────────┐
                                                          │ Context builder│
                                                          └───────────────┘
                                                                  │
                                                                  ▼
                                                       ┌────────────────────┐
                                                       │ Qwen/Qwen3-0.6B    │
                                                       │ (answer generation)│
                                                       └────────────────────┘
                                                                  │
                                                                  ▼
                                                       StudentSearchResultsAndAnswer
                                                            (JSON output)
```
 
- **Indexer** — walks `data/raw/`, chunks each file, and persists the index under
  `data/processed/`.
- **Retriever** — loads the index and scores chunks against a query using a lexical
  method, returning `MinimalSource` locations (`file_path`, character range).
- **Generator** — builds a prompt from the retrieved sources and calls
  `Qwen/Qwen3-0.6B` to produce a grounded answer.
- **CLI** (`python-fire`) — wires the three components together and exposes
  `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate`.
- **Data models** (`pydantic`) — validate everything passed between stages
  (`MinimalSource`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`,
  `StudentSearchResultsAndAnswer`, etc.).
<!-- TODO: adjust this section once your architecture is finalized —
     e.g. note any bonus components (embeddings, hybrid search, HTTP API) here. -->
 
## Chunking Strategy
 
Two distinct chunking strategies are implemented, since code and prose don't break
apart the same way:
 
- **Python code chunking** — <!-- TODO: describe your approach, e.g. AST-based
  splitting on function/class boundaries, falling back to fixed-size windows for very
  long definitions. -->
- **Markdown / text chunking** — <!-- TODO: describe your approach, e.g. splitting on
  headers/paragraphs with a sliding window and overlap. -->
Chunk size is capped at `--max_chunk_size` (default **2000 characters**, matching the
moulinette's `max_context_length`). <!-- TODO: report how smaller chunk sizes affected
your recall@k. -->
 
## Retrieval Method
 
<!-- TODO: state which method you implemented (BM25 and/or TF-IDF) and why. -->
 
- **Method**: BM25 / TF-IDF *(pick one or describe both)*
- **Ranking**: top-k chunks are returned per query, scored by lexical relevance to the
  question.
- **Source matching**: each result is a `(file_path, first_character_index,
  last_character_index)` triple, with `file_path` matching the ingested corpus path
  exactly (e.g. `data/raw/vllm-0.10.1/docs/features/lora.md`).
## Performance Analysis
 
<!-- TODO: replace with your actual measured numbers -->
 
| Metric | Target | Result |
|---|---|---|
| Indexing time | ≤ 5 min | — |
| Retrieval throughput (200 questions) | ≤ 90 s | — |
| Recall@5 (docs) | ≥ 80% | — |
| Recall@5 (code) | ≥ 50% | — |
 
| k | Recall@k |
|---|---|
| 1 | — |
| 3 | — |
| 5 | — |
| 10 | — |
 
## Design Decisions
 
<!-- TODO: fill in your actual choices, e.g.: -->
 
- Why BM25 over TF-IDF (or vice versa).
- How chunk size / overlap was tuned.
- How the prompt to `Qwen/Qwen3-0.6B` was structured to stay grounded.
- Any extensions made to the base pydantic models.
## Challenges Faced
 
<!-- TODO: document real difficulties and how you solved them, e.g.: -->
 
- Handling files that exceed `max_chunk_size`.
- Balancing recall on docs vs. code questions.
- Keeping the model's answers grounded despite its reasoning limits.
## Example Usage
 
```bash
$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0:1352]
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1936:3170]
...
```
 
<!-- TODO: add a real `answer` example once generation is working. -->
 
## Resources
 
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Fire documentation](https://github.com/google/python-fire)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
**AI usage**: <!-- TODO: be specific and honest, e.g. — "Claude was used to scaffold
the CLI structure and Makefile, to debug mypy/flake8 issues, and to draft this README.
All retrieval/chunking logic and prompt design were implemented and understood by the
author; AI-generated suggestions were reviewed and tested before inclusion." -->
