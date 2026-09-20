from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="Hello, explain Python in simple terms."
)

print(response.output_text)