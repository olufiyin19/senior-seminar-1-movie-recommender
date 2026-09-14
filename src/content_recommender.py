import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/ml-latest-small-master/movies.csv")

# print(movies[["title", "genres"]].head())

tfidf = TfidfVectorizer(token_pattern=r"[^|]+")

# Check the size of the genre matrix
genre_matrix = tfidf.fit_transform(movies["genres"])
# print("Genre matrix shape:", genre_matrix.shape)

similarity_matrix = cosine_similarity(genre_matrix)

movie_title = "Toy Story (1995)"
movie_index = movies[movies["title"] == movie_title].index[0]

similarity_scores = list(enumerate(similarity_matrix[movie_index]))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
top_matches = similarity_scores[1:6]

print(f"\nRecommendations similar to {movie_title}:")

for movie_index, score in top_matches:
    print(movies.iloc[movie_index]["title"], "-", score)