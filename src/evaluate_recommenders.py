import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the MovieLens dataset
ratings = pd.read_csv("data/ml-latest-small-master/ratings.csv")
movies = pd.read_csv("data/ml-latest-small-master/movies.csv")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

# Split each user's ratings into 80% training and 20% testing
train_parts = []
test_parts = []

for user_id, user_ratings in ratings.groupby("userId"):
    shuffled = user_ratings.sample(frac=1, random_state=42)
    split_index = int(len(shuffled) * 0.8)

    train_parts.append(shuffled.iloc[:split_index])
    test_parts.append(shuffled.iloc[split_index:])

train_ratings = pd.concat(train_parts, ignore_index=True)
test_ratings = pd.concat(test_parts, ignore_index=True)

print("\nTraining ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))

def make_reduced_training_set(train_ratings, availability, seed):
    if availability == 1.00:
        return train_ratings.copy()

    reduced_parts = []

    for user_id, user_ratings in train_ratings.groupby("userId"):
        sampled_ratings = user_ratings.sample(
            frac=availability,
            random_state=seed
        )
        reduced_parts.append(sampled_ratings)

    return pd.concat(reduced_parts, ignore_index=True)

# Define training-data availability levels and random seeds
availability_levels = [0.20, 0.40, 0.60, 0.80, 1.00]
subsample_seeds = [42, 43, 44, 45, 46]

# Generate the same training conditions as experiment.py
training_conditions = {}

for availability in availability_levels:
    seeds = subsample_seeds if availability < 1.00 else [42]

    for seed in seeds:
        training_conditions[(availability, seed)] = make_reduced_training_set(
            train_ratings,
            availability,
            seed
        )

print("\nTraining conditions prepared:", len(training_conditions))

# Find movies available across all training conditions
common_movie_ids = set(train_ratings["movieId"].unique())

for condition_ratings in training_conditions.values():
    common_movie_ids &= set(condition_ratings["movieId"].unique())

print("Movies available in all conditions:", len(common_movie_ids))

# Find users eligible across all training conditions
common_user_ids = set(train_ratings["userId"].unique())

for condition_ratings in training_conditions.values():
    ratings_per_user = condition_ratings.groupby("userId").size()

    eligible_user_ids = set(
        ratings_per_user[ratings_per_user >= 5].index
    )

    common_user_ids &= eligible_user_ids

# Restrict test ratings to the common movie catalog
common_test_ratings = test_ratings[
    test_ratings["movieId"].isin(common_movie_ids)
].copy()

# Keep users with at least one relevant test movie
users_with_relevant_test_movies = set(
    common_test_ratings.loc[
        common_test_ratings["rating"] >= 4.0,
        "userId"
    ]
)

common_user_ids &= users_with_relevant_test_movies

print("Users eligible in all conditions:", len(common_user_ids))

# Keep test ratings only for the fixed evaluation users
common_test_ratings = common_test_ratings[
    common_test_ratings["userId"].isin(common_user_ids)
].copy()

print("Fixed evaluation users:", len(common_user_ids))
print("Fixed evaluation test ratings:", len(common_test_ratings))

# Build genre-based movie features
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")

genre_matrix = tfidf.fit_transform(
    movies["genres"].fillna("")
)

print("Genre feature matrix shape:", genre_matrix.shape)

# Map each movie ID to its row in the genre feature matrix
movie_id_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movies["movieId"])
}

print("Movies indexed:", len(movie_id_to_index))

def get_content_scores(user_id, current_train_ratings):
    # Find movies this user rated highly in the training data
    liked_movie_ids = current_train_ratings.loc[
        (current_train_ratings["userId"] == user_id) &
        (current_train_ratings["rating"] >= 4.0),
        "movieId"
    ]

    # Match movie IDs to their genre matrix rows
    liked_indices = [
        movie_id_to_index[movie_id]
        for movie_id in liked_movie_ids
        if movie_id in movie_id_to_index
    ]

    # Handle users without any highly rated training movies
    if not liked_indices:
        return np.zeros(len(movies))

    # Build the user's genre preference profile
    user_profile = np.asarray(
        genre_matrix[liked_indices].mean(axis=0)
    )

    # Calculate similarity between the user and every movie
    content_scores = cosine_similarity(
        user_profile,
        genre_matrix
    ).flatten()

    return content_scores


def ndcg_at_k(recommended_movie_ids, relevant_movie_ids, k=10):
    """Calculate NDCG@K using binary relevance."""
    discounts = np.log2(np.arange(2, k + 2))

    relevance = np.array([
        1.0 if movie_id in relevant_movie_ids else 0.0
        for movie_id in recommended_movie_ids[:k]
    ])

    dcg = np.sum(relevance / discounts[:len(relevance)])

    ideal_count = min(len(relevant_movie_ids), k)

    if ideal_count == 0:
        return 0.0

    idcg = np.sum(1.0 / discounts[:ideal_count])
    return float(dcg / idcg)


def normalize_scores(scores):
    """Min-max normalization with protection against constant scores."""
    scores = np.asarray(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if np.isclose(maximum, minimum):
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def evaluate_condition(current_train_ratings, k=10):
    """Evaluate all three methods under one training condition."""

    user_movie_matrix = current_train_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )

    n_components = min(
        50,
        min(user_movie_matrix.shape) - 1
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    user_factors = svd.fit_transform(user_movie_matrix)
    predicted_ratings = svd.inverse_transform(user_factors)

    collaborative_movie_ids = user_movie_matrix.columns.to_numpy()

    # Prepare fixed catalog in a consistent order
    candidate_movie_ids = np.array(sorted(common_movie_ids))

    collaborative_indices = user_movie_matrix.columns.get_indexer(
        candidate_movie_ids
    )

    content_indices = np.array([
        movie_id_to_index[movie_id]
        for movie_id in candidate_movie_ids
    ])

    # Build lookups once per condition
    user_row_lookup = {
        user_id: index
        for index, user_id in enumerate(user_movie_matrix.index)
    }

    original_seen = (
        train_ratings.groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    relevant_test = (
        common_test_ratings[
            common_test_ratings["rating"] >= 4.0
        ]
        .groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    method_scores = {
        "content": [],
        "collaborative": [],
        "hybrid": []
    }

    for user_id in sorted(common_user_ids):
        user_row = user_row_lookup[user_id]

        collaborative_scores = predicted_ratings[
            user_row, collaborative_indices
        ]

        all_content_scores = get_content_scores(
            user_id,
            current_train_ratings
        )

        content_scores = all_content_scores[content_indices]

        # Normalize scores for the hybrid method
        collaborative_normalized = normalize_scores(
            collaborative_scores
        )

        content_normalized = normalize_scores(
            content_scores
        )

        hybrid_scores = (
            0.5 * collaborative_normalized
            + 0.5 * content_normalized
        )

        # Exclude every movie rated in the original training split
        unseen_mask = ~np.isin(
            candidate_movie_ids,
            list(original_seen.get(user_id, set()))
        )

        unseen_movie_ids = candidate_movie_ids[unseen_mask]

        scores_by_method = {
            "content": content_scores[unseen_mask],
            "collaborative": collaborative_scores[unseen_mask],
            "hybrid": hybrid_scores[unseen_mask]
        }

        relevant_movies = relevant_test.get(user_id, set())

        for method, scores in scores_by_method.items():
            top_indices = np.argsort(-scores, kind="stable")[:k]
            top_movie_ids = unseen_movie_ids[top_indices]

            user_ndcg = ndcg_at_k(
                top_movie_ids,
                relevant_movies,
                k=k
            )

            method_scores[method].append(user_ndcg)

    return {
        method: {
            "mean_ndcg": float(np.mean(scores)),
            "users_evaluated": len(scores)
        }
        for method, scores in method_scores.items()
    }


# Run all recommendation methods under the same conditions
comparison_results = []

for (availability, seed), current_train_ratings in training_conditions.items():
    print(
        f"Evaluating availability={availability:.0%}, seed={seed}",
        flush=True
    )

    condition_results = evaluate_condition(current_train_ratings)

    for method, metrics in condition_results.items():
        comparison_results.append({
            "availability": int(availability * 100),
            "seed": seed,
            "method": method,
            "users_evaluated": metrics["users_evaluated"],
            "mean_ndcg": metrics["mean_ndcg"]
        })


comparison_df = pd.DataFrame(comparison_results)

comparison_df.to_csv(
    "results/recommender_comparison_results.csv",
    index=False
)

print("\nComparison summary:")
print(
    comparison_df.groupby(
        ["availability", "method"]
    )["mean_ndcg"].agg(["mean", "std"]).to_string()
)

print("\nTotal experiment results:", len(comparison_df))
print("Results saved to results/recommender_comparison_results.csv")
