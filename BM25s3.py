import bm25s
from bm25s.hf import BM25HF

# Load the index
retriever = BM25HF.load_from_hub(f"{BM25}/bm25s-animals", load_corpus=True)

# You can retrieve now
query = "a cat is a feline"

docs, scores = retriever.retrieve(bm25s.tokenize(query), k=2)
