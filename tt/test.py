import bm25s


chunks = [
    "is python python data",  # Chunk 0
    "python code",         # Chunk 1
    "data model python"           # Chunk 2
]

idf = len(chunks) / len(all(x for x in chunks if "python" in x))
idf = 3/2

len(idf(3/3))
# Step 1: Tokenization
tokens = bm25s.tokenize(chunks, stopwords=None)
print(tokens)

# Step 2: Create BM25
retriever = bm25s.BM25()

# Step 3: Build index
retriever.index(tokens)

# Step 4: Save index
retriever.save("tt/bm25_index", corpus=chunks)