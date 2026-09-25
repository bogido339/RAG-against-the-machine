from transformers import AutoModelForCausalLM, AutoTokenizer
from src.retriever import Retriever


class QwenRAG:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, user_input):
        messages = [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response


class Generator:
    def __init__(self):
        self.model = QwenRAG()

    def get_prompt(self, query, top_chunks):

        return (f"""Answer the question based only on this context:

Context:
{chr(10).join(top_chunks)}

Question: {query}

Answer:""")


    def answer(self, query, k):
        retriver = Retriever()
        top_k = retriver.search(query, k)
        
        prompt = self.get_prompt(query, [chunk["content"] for chunk in top_k])

        respons = self.model.generate_response(prompt)
        print(respons)
        print("##" * 40)
        for k in top_k:
            print(k)
            print("--" * 40)
            print("")
    
