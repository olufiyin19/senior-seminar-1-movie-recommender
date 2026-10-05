import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Load MovieLens ratings data
ratings = pd.read_csv("data/ml-latest-small-master/ratings.csv")
movies = pd.read_csv("data/ml-latest-small-master/movies.csv")

print(ratings.head())
print("\nNumber of ratings:", len(ratings))
print("Number of users:", ratings["userId"].nunique())
print("Number of movies:", ratings["movieId"].nunique())

# Create user-movie ratings matrix
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0
)

print("\nUser-movie matrix shape:", user_movie_matrix.shape)
print(user_movie_matrix.iloc[:5, :5])

# Apply TruncatedSVD for collaborative filtering
svd = TruncatedSVD(n_components=50, random_state=42)

user_factors = svd.fit_transform(user_movie_matrix)

print("\nSVD user factors shape:", user_factors.shape)
print("Explained variance:", svd.explained_variance_ratio_.sum())

# Reconstruct predicted preference scores
predicted_ratings = svd.inverse_transform(user_factors)

print("\nPredicted ratings matrix shape:", predicted_ratings.shape)
print("First user's first 5 predicted scores:")
print(predicted_ratings[0, :5])
# Print movies data
print("\nMovies data:")
print(movies.head())

# Generate recommendations for a sample user
selected_user = 1

# Find this user's row in the matrix
user_index = user_movie_matrix.index.get_loc(selected_user)

# Get the user's predicted scores
user_scores = predicted_ratings[user_index]

# Get movies the user has already rated
rated_movie_ids = set(
    ratings.loc[ratings["userId"] == selected_user, "movieId"]
)

# Pair each movie ID with its predicted score
recommendations = pd.DataFrame({
    "movieId": user_movie_matrix.columns,
    "predicted_score": user_scores
})

# Remove movies the user has already rated
recommendations = recommendations[
    ~recommendations["movieId"].isin(rated_movie_ids)
]

# Sort from highest to lowest predicted score
recommendations = recommendations.sort_values(
    "predicted_score",
    ascending=False
).head(5)

# Add movie titles
recommendations = recommendations.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

print(f"\nTop 5 collaborative recommendations for User {selected_user}:")
print(recommendations[["title", "predicted_score"]].to_string(index=False))