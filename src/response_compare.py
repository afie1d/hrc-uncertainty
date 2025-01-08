import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw

nlp = spacy.load("en_core_web_sm") 

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



# ***** TESTING *****

t1 = "I will pick up the red block, then the green block"
t2 = "I will pick up the green block, then the yellow block"
t3 = "I will pick up the screwdriver, then the hammer"
t4 = "I will not pick up the screwdriver or the hammer"
t5 = "The headphones fit nicely around my ears"

def test_methods(texts1, texts2):
    for text1, text2 in zip(texts1, texts2):
        ss_measure = semantic_similarity(text1, text2, nlp)
        cs_tfdif_measure = cosine_similarity_tfidf(text1, text2)
        dtw_measure = dtw_similarity(text1, text2)
        print("----------------")
        print(text1, '\n', text2)
        print("Semantic similarity:", ss_measure)
        print("TFIDF Similarity:", cs_tfdif_measure)

a = [t1, t3, t5, t1]
b = [t2, t4, t3, t4]
test_methods(a, b)