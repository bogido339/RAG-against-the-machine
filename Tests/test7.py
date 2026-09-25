from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chunks = [
    "Retrieval-Augmented Generation (RAG) architecture combines a dense retriever with a sequence-to-sequence generator to produce contextually accurate responses.",
    "Vector embeddings are numerical representations of text chunks captured within a high-dimensional vector space, enabling semantic similarity search via cosine distance.",
    "Document chunking strategies typically involve splitting large files into smaller overlapping segments to ensure context preservation and precise retrieval token limits.",
    "Pydantic models are extensively used in modern Python pipelines to enforce strict data validation, runtime type checking, and robust schema parsing for structured outputs.",
    "Command-Line Interface frameworks like Python Fire automatically transform existing modules, classes, and functions into fully functional CLI tools without extra boilerplate code.",
    "Indexing documents efficiently requires building optimized vector indexes using algorithms such as Hierarchical Navigable Small World graphs for fast approximate nearest neighbor lookups.",
    "Context window limitations in Large Language Models necessitate careful pruning and reranking of retrieved documents to avoid hallucinations and maximize informational relevance.",
    "Hybrid search techniques integrate traditional keyword-based BM25 lexical matching with dense vector retrieval to capture both exact terminology and deep semantic intent.",
    "Python dependency management tools help isolate virtual environments, track locked package versions, and streamline reproducible deployments across different development machines.",
    "Evaluation frameworks for RAG systems measure metrics like faithfulness, answer relevance, and context precision to quantify the overall reliability of generated answers."
]

question = "What are the primary advantages of combining keyword-based BM25 lexical matching with dense vector retrieval in a RAG pipeline?"

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(chunks + [question])

scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

best_5 = scores.argsort()[-5:][::-1]

for i in best_5:
    print(scores[i], chunks[i])