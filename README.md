*This project has been created as part of the 42 curriculum by mbougajd.*

# RAG Against the Machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) codebase without retraining or fine-tuning a language model.

The pipeline:

1. **Indexes** Python source files and Markdown documentation into searchable text chunks.
2. **Retrieves** relevant chunks using BM25 lexical search.
3. **Augments** the language model's prompt with retrieved source content.
4. **Generates** a natural-language answer using `Qwen/Qwen3-0.6B`.

Retrieval quality is evaluated using **Recall@k**. The stages are accessible through a command-line interface (CLI).

## Instructions

### Prerequisites

- Python compatible with the project's `pyproject.toml` (check `requires-python`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- The vLLM source corpus in `data/raw/vllm-0.10.1/`.

### Installation

```bash
git clone <your-repo-url>
cd rag-against-the-machine
make install
```

`make install` runs `uv sync` to create the virtual environment and install dependencies from `pyproject.toml` and `uv.lock`.

### Makefile commands

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync`. |
| `make run` | Run the default CLI command (`index`). |
| `make run FLAG=search` | Select a different CLI command; supply additional arguments directly through `uv run`. |
| `make debug` | Run the project with Python's debugger (`pdb`). |
| `make clean` | Remove Python caches and build artifacts. |
| `make lint` | Run Flake8 and Mypy checks. |
| `make lint-strict` | Run Flake8 and strict Mypy checks. |

### Running the pipeline

Run the CLI as `uv run python -m src <command>`.

**1. Build the index**

```bash
uv run python -m src index --max_chunk_size 2000
```

**2. Retrieve sources for one question**

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

**3. Retrieve sources for a dataset**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**4. Generate an answer for one question**

```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

**5. Generate answers for a dataset**

```bash
uv run python -m src answer_dataset \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/output/search_results_and_answer/UnansweredQuestions
```

**6. Evaluate retrieval against labeled answers**

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

For code questions, replace `dataset_docs_public.json` with `dataset_code_public.json` in the corresponding paths.

> **Note:** Keep each trailing `\` as the last character of its shell line. The next line is part of the same command.

## System Architecture

```text
vLLM source files (data/raw/vllm-0.10.1/)
                  |
                  v
          Indexer + chunking
                  |
                  v
     chunks.json + BM25 index
                  |
                  v
           BM25 Retriever <--- User question
                  |
                  v
         Top-k source locations
                  |
                  v
          Source text loading
                  |
                  v
           Prompt construction
                  |
                  v
            Qwen3-0.6B
                  |
                  v
           Generated answer
```

- **Indexer:** Reads source files, splits their content into chunks, and saves chunk metadata and the BM25 index under `data/processed/`.
- **Retriever:** Loads the saved index, ranks chunks by lexical relevance, and returns source locations containing file paths and character offsets.
- **Generator:** Loads retrieved text, builds a context-aware prompt, and generates an answer with Qwen3-0.6B.
- **CLI:** Uses `python-fire` to expose indexing, searching, answer generation, and dataset operations.
- **Data models:** Use Pydantic to validate structured inputs and outputs, including `MinimalSource`, `StudentSearchResults`, and `StudentSearchResultsAndAnswer`.

## Chunking Strategy

The indexer uses `RecursiveCharacterTextSplitter` with Markdown-oriented separators to split source content into manageable chunks.

- **Maximum chunk size:** 2,000 characters by default, configurable with `--max_chunk_size`.
- **Overlap:** 5% of the configured chunk size (100 characters when the chunk size is 2,000).
- **Source metadata:** Each chunk records `file_path`, `content`, `first_character_index`, and `last_character_index`.

Chunk overlap helps preserve context across adjacent chunks. This is a character-based strategy; it does not guarantee that every Python function or class remains intact.

## Retrieval Method

The project uses **BM25**, implemented with the `bm25s` Python package.

1. Tokenize the indexed chunks.
2. Build and save the BM25 index.
3. Tokenize the user's query.
4. Score and rank chunks by lexical relevance.
5. Return up to `k` matching source locations.

BM25 is a practical baseline because it does not require an embedding model, supports exact identifier matching, and is efficient for code and documentation search. Its main limitation is that it can miss semantically similar passages that use different vocabulary.

Each source location contains:

```json
{
  "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 0,
  "last_character_index": 1200
}
```

The values above illustrate the output format; they are not a recorded retrieval result.

## Performance Analysis

The following Recall@k results were measured on the project's public labeled datasets:

| k | Documentation (100 questions) | Code (100 questions) |
| ---: | ---: | ---: |
| 1 | 59.0% | 33.3% |
| 3 | 77.0% | 48.5% |
| 5 | **82.0%** | **55.6%** |
| 10 | 90.0% | 58.6% |

At `k = 5`, both reported recall targets were met: at least 80% for documentation and 50% for code.

| Additional metric | Target | Measured result |
| --- | --- | --- |
| Indexing time | ≤ 5 minutes | Not measured |
| Retrieval time for 200 questions | ≤ 90 seconds | Not measured |

**Interpretation:** BM25 retrieves documentation passages more reliably than code passages on these datasets. Increasing `k` improves coverage but increases the amount of retrieved material to process.

## Design Decisions

- **BM25 instead of embedding retrieval:** Keeps the pipeline lightweight and favors exact terms such as function names, configuration options, and API identifiers.
- **Recursive chunking:** Avoids indexing entire large files as single documents while keeping nearby text together where possible.
- **5% overlap:** Reduces the risk of losing information at chunk boundaries.
- **Source offsets:** Allow retrieved results to point back to specific spans in the original corpus.
- **Pydantic models:** Help catch invalid structured data between pipeline stages.
- **Separate search and generation:** Makes it possible to evaluate retrieval independently from language-model answer quality.

## Challenges and Limitations

- **Code retrieval:** Identifier-heavy and structurally complex code can be harder to retrieve than explanatory documentation.
- **Chunk boundaries:** Character-based splitting can separate related code statements or documentation paragraphs.
- **Grounding:** Retrieved context improves answer relevance but does not guarantee factual correctness. Generated answers should be checked against their sources.
- **Local inference:** Model loading and answer generation may be slow on CPU-only systems.
- **Input and file errors:** Missing indices, malformed datasets, invalid CLI arguments, and filesystem permissions require explicit validation and useful error messages.

## Example Usage

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

The search command returns source locations from the indexed vLLM corpus. The `answer` command retrieves source text and passes it to Qwen3-0.6B to generate a response.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*.
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [BM25S](https://github.com/xhluca/bm25s)
- [Python Fire documentation](https://github.com/google/python-fire)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

## AI Usage

AI was used only as a helper for documentation, formatting, explanations, and debugging suggestions