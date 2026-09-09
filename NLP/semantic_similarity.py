from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is a popular programming language.",
    "Neural networks are used in deep learning.",
    "Transformers use self attention to process sequences.",
    "Football is a popular sport played around the world.",
    "Machine learning models learn patterns from data."
]

doc_embedding= model.encode(documents)


print("Number of documents:", len(documents))
print("Embedding shape:", doc_embedding.shape)


query = "How do neural networks work?"

query_em=model.encode(query)

similarities = cosine_similarity(
    [query_em],
    doc_embedding
)[0]


for document, score in zip(documents, similarities):

    print(f"\nScore: {score:.4f}")
    print(document)


# ============================================================
# 8. FIND MOST RELEVANT DOCUMENT
# ============================================================

best_index = np.argmax(similarities)

print("\n==============================")
print("MOST RELEVANT DOCUMENT")
print("==============================")

print(documents[best_index])
print("Score:", similarities[best_index])

