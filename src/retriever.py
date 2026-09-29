from typing import List, Dict, Any
import json
import bm25s


class Retriever:
    """Loads the saved BM25 index and performs similarity search."""

    def __init__(
        self, 
        chunks_path: str = "data/processed/chunks.json", 
        bm25_index: str = "data/processed/bm25_index"
    ):
        self.chunks_path = chunks_path
        self.bm25_index = bm25_index
        
        # try and except don't forget!!
        self.retriever = bm25s.BM25.load(
            self.bm25_index,
            load_corpus=True
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search the corpus for the most relevant chunks given a query."""
        top_k_chunks = []

        results, scores = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=top_k
        )

        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                for res in results[0]:
                    for chunk in chunks:
                        if chunk["content"] == res["text"]:
                            top_k_chunks.append(chunk)
        except FileNotFoundError:
            print('Wornnign: run first this commond: "uv run python -m src index 2000"')
        except PermissionError:
            print("Wornning: permission dinay write the commond: 'chmod 777 file'")

        return top_k_chunks

    def topk_search(self, query, top_k):

        for chunk in self.search(query, top_k):
            print(
                f"{chunk["file_path"]} [{chunk["first_character_index"]}:"
                f"{chunk["last_character_index"]}]"
            )


class DatasetRetriever():
    def __init__(self):
        self.retriever = Retriever()

    def search_dataset(self, dataset_path, save_directory, k):
        data = []
        res = {}

        try:
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
        except:
            print("wornning: try again")

        res.update({"search_results": data})
        res.update({"k": k})
        
        try:
            with open(save_directory, "w", encoding=f"utf-8") as f:
                json.dump(res, f, indent=4)
        except:
            print("save_directory path not good input good path and try again")
