# Install the library if you haven't: pip install -U langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "LangChain is a powerful framework for developing applications powered by language models. " \
       "Text splitters in LangChain help break large documents into smaller pieces for processing."

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

# Split the plain text into chunks
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
