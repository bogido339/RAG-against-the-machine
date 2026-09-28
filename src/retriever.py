from typing import List, Dict, Any
import json
import bm25s


class Retriever:
    """Loads the saved TF-IDF index and performs similarity search."""

    def __init__(
        self, 
        chunks_path: str = "data/processed/chunks.json", 
        bm25_index: str = "data/bm25_index"
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        self.retriever = bm25s.BM25.load(
            bm25_index,
            load_corpus=True
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the corpus for the most relevant chunks given a query."""
        top_k_chunks = []

        results, scores = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=top_k
        )
        for res in results[0]:
            for chunk in self.chunks:
                if chunk["content"] == res["text"]:
                    top_k_chunks.append(chunk)

        return top_k_chunks
    
class DatasetRetriever():
    def __init__(self):
        self.retriever = Retriever()

    def search_dataset(self, dataset_path, save_directory, k):
        data = []
        res = {}

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

            for query in dataset["rag_questions"]:
                question_id = query["question_id"]
                question = query["question"]
                
                dic = {
                    "question_id": question_id,
                    "question": question,
                    "retrieved_sources": self.retriever.search(question, k)
                }

                data.append(dic)
        res.update({"search_results": data})
        res.update({"k": k})

        with open(save_directory, "w", encoding=f"utf-8") as f:
            json.dump(res, f, indent=4)