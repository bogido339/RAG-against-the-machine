from sklearn.metrics.pairwise import cosine_similarity
import json
import joblib
from typing import List, Dict, Any


class Retriever:
    """Loads the saved TF-IDF index and performs similarity search."""

    def __init__(
        self, 
        chunks_path: str = "data/chunks.json", 
        vectorizer_path: str = "data/tfidf_vectorizer.pkl", 
        matrix_path: str = "data/tfidf_matrix.pkl"
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        self.vectorizer = joblib.load(vectorizer_path)
        self.tfidf_matrix = joblib.load(matrix_path)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the corpus for the most relevant chunks given a query."""
        query_vector = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "file_path": self.chunks[idx]["file_path"],
                "content": self.chunks[idx]["content"]
            })
            
        return results
