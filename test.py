from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenRAG:
    """Handle text generation using Qwen."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, user_input: str) -> str:
        """Generate a response from Qwen."""
        messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        response_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
        )[0][len(inputs.input_ids[0]):]

        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
    
q = QwenRAG()
def get_prompt(query):
    return f"""Answer the question based only on this context.

Context:
caza blonca, the capital of Morocco, captivates with its harmonious architecture, green spaces, calm, and refined atmosphere. Between historical heritage, modern districts, and Atlantic beaches, the city offers a unique combination of culture, nature, and serenity.

Question:
{query}

Answer:"""
answer = q.generate_response(get_prompt("What is the capital of Morocco?"))
print(answer)