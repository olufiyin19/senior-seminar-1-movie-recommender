import pandas as pd
import matplotlib.pyplot as plt

movies = pd.read_csv("data/ml-latest-small-master/movies.csv")
ratings = pd.read_csv("data/ml-latest-small-master/ratings.csv")

print(movies.head())
print(ratings.head())

print("Number of movies:", len(movies))
print("Number of ratings:", len(ratings))
print("Number of users:", ratings["userId"].nunique())

print("Average rating:", ratings["rating"].mean())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())

ratings_per_user = ratings.groupby("userId").size()

print("\nRatings per user:")
print(ratings_per_user.describe())

rating_counts = ratings["rating"].value_counts().sort_index()

rating_counts.plot(kind="bar")

plt.title("MovieLens Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.tight_layout()
plt.savefig("results/rating_distribution.png")
plt.close()

# Create a histogram of ratings per user
ratings_per_user.plot(kind="hist", bins=30)

plt.title("Distribution of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")

plt.tight_layout()
plt.savefig("results/ratings_per_user.png")
plt.close()