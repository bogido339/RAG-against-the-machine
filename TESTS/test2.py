import bm25s
from typing import List, Any, Dict


def build_bm25_index(chunks: List[str],) -> None:
    """Build and save the BM25 index."""
    
    retriever = bm25s.BM25(corpus=chunks)

    retriever.index(bm25s.tokenize(chunks))

    retriever.save("hello_folder/bm25_test")

corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

build_bm25_index(corpus)