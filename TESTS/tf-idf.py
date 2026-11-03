# from sklearn.feature_extraction.text import TfidfVectorizer

# # Sample corpus
# documents = [
#     "The cat sat on the mat.",
#     "The dog played in the park.",
#     "Cats and dogs are great pets."
# ]

# # Initialize and transform the text into a matrix
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(documents)

# # View the learned vocabulary tokens
# print(vectorizer.get_feature_names_out())


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
documents = [
    "The cat sat on the mat.",
    "The dog played in the park.",
    "Cats and dogs are great pets."
]

words = set()
for doc in documents:
    words.update(doc.split())

print(words)

# Initialize and transform the text into a matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the sparse matrix into a Pandas DataFrame for clear visualization
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f"Doc {i+1}" for i in range(len(documents))]
)

# Display the TF-IDF table
print("--- TF-IDF Matrix Table ---")
print(df)