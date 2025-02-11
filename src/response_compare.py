from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_similarity(text1, text2, model):
    doc1 = model(text1)
    doc2 = model(text2)
    return doc1.similarity(doc2)

def cosine_similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


def pairwise_similarity(responses):
    diffs = []
    for i, r in enumerate(responses):
        for j in range(len(responses)):
            if i == j: continue
            diffs.append(cosine_similarity_tfidf(r, responses[j]))

    return diffs

def uncertainty_score(responses):
    sims = pairwise_similarity(responses)
    return 1 / np.mean(sims)
