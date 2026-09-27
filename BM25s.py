# pip install bm25s
import bm25s

corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

# Tokenize corpus
corpus_tokens = bm25s.tokenize(corpus)

# Create and index BM25 model
retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

# Query
query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query)

# Get top 2 results
results, scores = retriever.retrieve(query_tokens, k=2)

# Print results
for i, (doc, score) in enumerate(zip(results[0], scores[0])):
    print(f"Rank {i + 1} (score: {score:.2f}): {doc}")