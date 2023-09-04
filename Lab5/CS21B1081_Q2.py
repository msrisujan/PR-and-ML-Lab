import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

with open('doc1.txt', 'r') as file1:
    text1 = file1.read()

with open('doc2.txt', 'r') as file2:
    text2 = file2.read()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text1, text2])

vector1 = X[0].toarray().flatten()
vector2 = X[1].toarray().flatten()

dot_product = np.dot(vector1, vector2)
magnitude_vector1 = np.sqrt(np.sum(np.square(vector1)))
magnitude_vector2 = np.sqrt(np.sum(np.square(vector2)))

cosine_similarity = dot_product / (magnitude_vector1 * magnitude_vector2)

print("Cosine Similarity:", cosine_similarity)

print("Cosine distance:", 1 - cosine_similarity)