import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/movies.csv', encoding='latin-1', sep='\t', usecols=["title", "genres"])

df["genres"] = df["genres"].str.replace("|", " ").str.replace("-", "")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["genres"])
tfidf_matrix_dense = pd.DataFrame(vectorizer.fit_transform(df["genres"]).todense(), index=df['title'],
                                  columns=vectorizer.get_feature_names_out())  # dua ve matrix cho de nhin

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])
