from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """LangChain is a powerful framework for developing applications powered by language models.
It enables developers to chain together components like LLMs, prompts, and memory to create advanced conversational AI systems.
Text splitters in LangChain help break large documents into smaller pieces for processing."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=40*0.1,
)

chunks = splitter.split_text(text)

print("📄 Number of Chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk}")
