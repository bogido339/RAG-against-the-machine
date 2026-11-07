# documents = [
#     "python handles exceptions",
#     "python uses lists",                    # {
#     "exceptions happen during execution",   #   "python": {0: 1, 1: 1},
#     "lists store values"                    #   "handles": {0: 1},
#                                             #   "exceptions": {0: 1, 2: 1},
#                                             #   "uses": {1: 1},
#                                             #   "lists": {1: 1, 3: 1},
#                                             #   "happen": {2: 1},
#                                             #   "during": {2: 1},
#                                             #   "execution": {2: 1},
#                                             #   "store": {3: 1},
#                                             #   "values": {3: 1}
#                                             # }
# ]

# inverted_index = {}

# for doc_id, document in enumerate(documents):
#     tokens = document.split()

#     for token in tokens:
#         if token not in inverted_index:
#             inverted_index[token] = {}

#         if doc_id not in inverted_index[token]:
#             inverted_index[token][doc_id] = 0

#         inverted_index[token][doc_id] += 1

# print(inverted_index)




# import math
import bm25s

documents = [
    "python is handles exceptions",
    "python uses lists",
    "exceptions happen during execution",
    "lists store values"
]

corpus_tokens = bm25s.tokenize(documents)
print(corpus_tokens)

# query = "python exceptions"

# # Parameters
# k1 = 1.5
# b = 0.75

# # Tokenization
# tokenized_docs = [doc.split() for doc in documents]
# query_tokens = query.split()

# # Corpus statistics
# N = len(tokenized_docs)

# avgdl = sum(len(doc) for doc in tokenized_docs) / N


# def calculate_idf(term):
#     # df = sum(
#     #     1 for doc in tokenized_docs
#     #     if term in doc
#     # )

#     df = len(inverted_index[term])

#     return math.log(
#         1 + (N - df + 0.5) / (df + 0.5)
#     )


# def calculate_score(document, query_tokens):
#     score = 0.0

#     doc_length = len(document)

#     for term in set(query_tokens):
#         tf = document.count(term)

#         if tf == 0:
#             continue

#         idf = calculate_idf(term)

#         numerator = tf * (k1 + 1)

#         denominator = tf + k1 * (
#             1 - b + b * (doc_length / avgdl)
#         )

#         score += idf * (numerator / denominator)

#     return score


# for doc_id, document in enumerate(tokenized_docs):
#     score = calculate_score(document, query_tokens)

#     print(f"D{doc_id}: {score:.4f}")