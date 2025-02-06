<<<<<<< HEAD
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw

# nlp = spacy.load("en_core_web_sm") 

def semantic_similarity(text1, text2, model):
    doc1 = model(text1)
    doc2 = model(text2)
    return doc1.similarity(doc2)

def cosine_similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


def dtw_similarity(text1, text2):
    seq1 = text1.split()
    seq2 = text2.split()
    distance = dtw.distance(seq1, seq2) # error here for some reason?
    return 1 / (1 + distance)  # Invert distance to get similarity



def pairwise_diff(responses, sim_measure_function):
    diffs = []
    for i, r in enumerate(responses):
        for j in range(len(responses)):
            if i == j: continue
            diffs.append(sim_measure_function(r, responses[j]))

=======
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(text1, text2, model):
    doc1 = model(text1)
    doc2 = model(text2)
    return doc1.similarity(doc2)

def cosine_similarity_tfidf(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


def pairwise_diff(responses):
    diffs = []
    for i, r in enumerate(responses):
        for j in range(len(responses)):
            if i == j: continue
            diffs.append(cosine_similarity_tfidf(r, responses[j]))

>>>>>>> b2f4c8d (updated img adjust)
    return diffs