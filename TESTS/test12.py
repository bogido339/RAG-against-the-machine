from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
import bm25s


# separators = RecursiveCharacterTextSplitter.get_separators_for_language(
#     Language.MARKDOWN
# )

# print(separators)
# corpus = [
#     "Python is easy to learn.",     # 24
#     "Machine learning is fun.",    # 24
#     "Data science uses Python.",   # 25
#     "AI can understand text.",     # 23
#     "We build software systems.",  # 26
#     "Python supports many tools.", # 27
#     "Programming needs practice."  # 27
# ]

corpus = [
    "Python Python Python programming",
    "Python programming",
    "Machine learning"
]

retriever = bm25s.BM25()

attributes_and_methods = [attr for attr in dir(retriever) if not attr.startswith('_')]
# print(attributes_and_methods)

corpus_tokens = bm25s.tokenize(corpus)
# print("TYPE:", type(corpus_tokens))
# print(corpus_tokens.ids)
# print(corpus_tokens.vocab)
retriever.index(corpus_tokens)
retriever.save("data/processed/bm25_index", corpus=corpus)