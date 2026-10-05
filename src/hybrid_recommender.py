import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load MovieLens data
ratings = pd.read_csv("data/ml-latest-small-master/ratings.csv")
movies = pd.read_csv("data/ml-latest-small-master/movies.csv")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# Create content-based movie features using genres
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")

genre_matrix = tfidf.fit_transform(movies["genres"])

print("\nGenre feature matrix shape:", genre_matrix.shape)
print("Genre features:", tfidf.get_feature_names_out())

# Create user-movie ratings matrix for collaborative filtering
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0
)

print("\nUser-movie matrix shape:", user_movie_matrix.shape)

# Apply TruncatedSVD for collaborative filtering
svd = TruncatedSVD(n_components=50, random_state=42)

user_factors = svd.fit_transform(user_movie_matrix)

# Reconstruct predicted preference scores
predicted_ratings = svd.inverse_transform(user_factors)

print("\nSVD user factors shape:", user_factors.shape)
print("Predicted ratings matrix shape:", predicted_ratings.shape)

# Select a sample user
selected_user = 1

# Find the user's row in the ratings matrix
user_index = user_movie_matrix.index.get_loc(selected_user)

# Get collaborative predicted scores for this user
collaborative_scores = predicted_ratings[user_index]

# Get movies the user has already rated
rated_movie_ids = set(
    ratings.loc[ratings["userId"] == selected_user, "movieId"]
)

print(f"\nSelected user: {selected_user}")
print("Number of movies already rated:", len(rated_movie_ids))
print("Number of collaborative scores:", len(collaborative_scores))

# Get movies the user rated highly
liked_movie_ids = ratings.loc[
    (ratings["userId"] == selected_user) &
    (ratings["rating"] >= 4.0),
    "movieId"
]

print("\nNumber of movies User 1 rated 4 or higher:", len(liked_movie_ids))

# Find the rows of movies the user liked
liked_movie_indices = movies.index[
    movies["movieId"].isin(liked_movie_ids)
].tolist()

# Get the genre features for those movies
liked_genre_vectors = genre_matrix[liked_movie_indices]

# Build the user's content preference profile
user_content_profile = np.asarray(
    liked_genre_vectors.mean(axis=0)
)

print("\nLiked movies found in genre matrix:", len(liked_movie_indices))
print("User content profile shape:", user_content_profile.shape)

# Calculate content similarity scores for all movies
content_scores = cosine_similarity(
    user_content_profile,
    genre_matrix
).flatten()

print("\nNumber of content scores:", len(content_scores))
print("First 5 content scores:", content_scores[:5])

# Create DataFrame of content scores
content_score_df = pd.DataFrame({
    "movieId": movies["movieId"],
    "content_score": content_scores
})

# Create DataFrame of collaborative scores
collaborative_score_df = pd.DataFrame({
    "movieId": user_movie_matrix.columns,
    "collaborative_score": collaborative_scores
})

# Keep only movies that exist in both systems
hybrid_scores = collaborative_score_df.merge(
    content_score_df,
    on="movieId",
    how="inner"
)

print("\nAligned movies:", len(hybrid_scores))
print(hybrid_scores.head())

# Normalize both score types to a 0-1 range
hybrid_scores["collaborative_normalized"] = (
    hybrid_scores["collaborative_score"] -
    hybrid_scores["collaborative_score"].min()
) / (
    hybrid_scores["collaborative_score"].max() -
    hybrid_scores["collaborative_score"].min()
)

hybrid_scores["content_normalized"] = (
    hybrid_scores["content_score"] -
    hybrid_scores["content_score"].min()
) / (
    hybrid_scores["content_score"].max() -
    hybrid_scores["content_score"].min()
)

print("\nNormalized scores:")
print(
    hybrid_scores[
        ["movieId", "collaborative_normalized", "content_normalized"]
    ].head()
)

# Combine collaborative and content scores
alpha = 0.5

hybrid_scores["hybrid_score"] = (
    alpha * hybrid_scores["collaborative_normalized"]
    + (1 - alpha) * hybrid_scores["content_normalized"]
)

print("\nHybrid scores:")
print(
    hybrid_scores[
        [
            "movieId",
            "collaborative_normalized",
            "content_normalized",
            "hybrid_score"
        ]
    ].head()
)

# Remove movies the user has already rated
hybrid_recommendations = hybrid_scores[
    ~hybrid_scores["movieId"].isin(rated_movie_ids)
].copy()

# Sort by hybrid score and select the top 5
hybrid_recommendations = hybrid_recommendations.sort_values(
    "hybrid_score",
    ascending=False
).head(5)

# Add movie titles
hybrid_recommendations = hybrid_recommendations.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

print(f"\nTop 5 hybrid recommendations for User {selected_user}:")
print(
    hybrid_recommendations[
        [
            "title",
            "collaborative_normalized",
            "content_normalized",
            "hybrid_score"
        ]
    ].to_string(index=False)
)