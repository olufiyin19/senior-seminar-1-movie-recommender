import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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


WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def evaluate_weights(current_train_ratings, k=10):
    """Fit SVD once; score all five weights for each eligible user."""
    matrix = current_train_ratings.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    components = min(50, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=components, random_state=42)
    predictions = svd.inverse_transform(svd.fit_transform(matrix))

    candidates = np.array(sorted(common_movie_ids))
    collab_indices = matrix.columns.get_indexer(candidates)
    content_indices = np.array([movie_id_to_index[m] for m in candidates])
    user_rows = {u: i for i, u in enumerate(matrix.index)}
    original_seen = train_ratings.groupby("userId")["movieId"].agg(set).to_dict()
    relevant_test = (
        common_test_ratings[common_test_ratings["rating"] >= 4.0]
        .groupby("userId")["movieId"].agg(set).to_dict()
    )
    results = {w: [] for w in WEIGHTS}

    for user_id in sorted(common_user_ids):
        collab = predictions[user_rows[user_id], collab_indices]
        content = get_content_scores(user_id, current_train_ratings)[content_indices]
        # Match the existing midterm's normalization before excluding seen items.
        c_norm = normalize_scores(collab)
        g_norm = normalize_scores(content)
        unseen = ~np.isin(candidates, list(original_seen.get(user_id, set())))
        unseen_ids = candidates[unseen]
        relevant = relevant_test.get(user_id, set())

        for weight in WEIGHTS:
            scores = (weight * c_norm + (1.0 - weight) * g_norm)[unseen]
            top_ids = unseen_ids[np.argsort(-scores, kind="stable")[:k]]
            results[weight].append(ndcg_at_k(top_ids, relevant, k=k))

    return {weight: float(np.mean(values)) for weight, values in results.items()}


def main():
    Path("results").mkdir(exist_ok=True)
    rows = []
    for (availability, seed), current_train in training_conditions.items():
        print(f"Evaluating {availability:.0%} availability, seed {seed}", flush=True)
        scores = evaluate_weights(current_train)
        for weight, ndcg in scores.items():
            rows.append({
                "availability": int(availability * 100),
                "seed": seed,
                "collaborative_weight": weight,
                "content_weight": 1.0 - weight,
                "users_evaluated": len(common_user_ids),
                "mean_ndcg": ndcg,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/hybrid_weight_experiment_results.csv", index=False)
    summary = df.groupby(["availability", "collaborative_weight"])["mean_ndcg"].agg(
        mean="mean", std="std"
    ).reset_index()
    summary.to_csv("results/hybrid_weight_experiment_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for weight in WEIGHTS:
        subset = summary[summary["collaborative_weight"] == weight].sort_values("availability")
        ax.plot(subset["availability"], subset["mean"], marker="o", label=f"Collaborative weight {weight:.2f}")
    ax.set(xlabel="Training data availability (%)", ylabel="Mean NDCG@10",
           title="Hybrid weighting sensitivity across data availability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/hybrid_weight_comparison.png", dpi=180)
    plt.close(fig)

    # Check the two key baseline conditions against the previous experiment.
    baseline_path = Path("results/recommender_comparison_results.csv")
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        for weight, method in [(0.5, "hybrid"), (1.0, "collaborative")]:
            actual = df[df["collaborative_weight"] == weight][
                ["availability", "seed", "mean_ndcg"]
            ]
            expected = baseline[baseline["method"] == method][
                ["availability", "seed", "mean_ndcg"]
            ]
            matched = actual.merge(expected, on=["availability", "seed"],
                                   suffixes=("_new", "_old"), validate="one_to_one")
            assert len(matched) == 21, f"Missing baseline rows for {method}"
            assert np.allclose(matched["mean_ndcg_new"], matched["mean_ndcg_old"], atol=1e-10), (
                f"Baseline mismatch for {method}"
            )
            print(f"PASS: {method} baseline matches all 21 conditions")
    else:
        print("Baseline CSV not found; skipping automatic baseline comparison")

    print("\nMean NDCG@10 by availability and collaborative weight:")
    print(summary.pivot(index="availability", columns="collaborative_weight", values="mean").round(6))
    print(f"\nSaved {len(df)} evaluations, summary CSV, and comparison graph.")


if __name__ == "__main__":
    main()
