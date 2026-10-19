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


# Week 8–9 experiment: vary latent dimensionality while holding all
# evaluation users, candidate movies, and held-out ratings fixed.
COMPONENT_COUNTS = [10, 25, 50, 75, 100]
CANDIDATES = np.array(sorted(common_movie_ids))
ORIGINAL_SEEN = train_ratings.groupby("userId")["movieId"].agg(set).to_dict()
RELEVANT = (common_test_ratings[common_test_ratings["rating"] >= 4.0]
            .groupby("userId")["movieId"].agg(set).to_dict())


def ndcg_at_k(recommended, relevant, k=10):
    gains = np.array([movie_id in relevant for movie_id in recommended[:k]], dtype=float)
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
    ideal = min(len(relevant), k)
    if ideal == 0:
        return 0.0
    return float(dcg / np.sum(1.0 / np.log2(np.arange(2, ideal + 2))))


results = []
for (availability, seed), condition_ratings in training_conditions.items():
    print(f"Evaluating {availability:.0%} availability, seed {seed}", flush=True)
    matrix = condition_ratings.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    indices = matrix.columns.get_indexer(CANDIDATES)
    if (indices < 0).any():
        raise RuntimeError("A candidate movie is missing from the training matrix")
    user_lookup = {uid: i for i, uid in enumerate(matrix.index)}
    if not common_user_ids.issubset(user_lookup):
        raise RuntimeError("An evaluation user is missing from the training matrix")
    maximum = min(matrix.shape) - 1

    for requested_components in COMPONENT_COUNTS:
        if requested_components > maximum:
            raise RuntimeError(
                f"Cannot fit {requested_components} components; maximum is {maximum}"
            )
        svd = TruncatedSVD(n_components=requested_components, random_state=42)
        factors = svd.fit_transform(matrix)
        predictions = svd.inverse_transform(factors)
        user_ndcgs = []
        for uid in sorted(common_user_ids):
            unseen = ~np.isin(CANDIDATES, list(ORIGINAL_SEEN.get(uid, set())))
            movie_ids = CANDIDATES[unseen]
            scores = predictions[user_lookup[uid], indices][unseen]
            top = np.argsort(-scores, kind="stable")[:10]
            user_ndcgs.append(ndcg_at_k(movie_ids[top], RELEVANT.get(uid, set())))
        results.append({
            "availability": int(availability * 100),
            "seed": seed,
            "n_components": requested_components,
            "users_evaluated": len(user_ndcgs),
            "mean_ndcg": float(np.mean(user_ndcgs)),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        })

out = Path("results")
out.mkdir(exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(out / "svd_component_experiment_results.csv", index=False)
summary = (df.groupby(["availability", "n_components"], as_index=False)
           .agg(mean_ndcg=("mean_ndcg", "mean"),
                std_ndcg=("mean_ndcg", "std"),
                mean_explained_variance=("explained_variance_ratio_sum", "mean"),
                runs=("seed", "size")))
summary.to_csv(out / "svd_component_experiment_summary.csv", index=False)

# Verify the 50-component condition against the previously published baseline.
baseline = pd.read_csv(out / "recommender_comparison_results.csv")
baseline = baseline[baseline["method"] == "collaborative"]
merged = df[df["n_components"] == 50].merge(
    baseline[["availability", "seed", "mean_ndcg"]],
    on=["availability", "seed"], how="outer", suffixes=("_new", "_old"),
    indicator=True
)
if len(merged) != 21 or not (merged["_merge"] == "both").all() or not np.allclose(
    merged["mean_ndcg_new"], merged["mean_ndcg_old"], atol=1e-10, rtol=0
):
    raise AssertionError("50-component baseline mismatch; inspect results before reporting")
print("PASS: 50-component baseline matches all 21 previous conditions")

plt.figure(figsize=(9, 5.5))
for n, part in summary.groupby("n_components"):
    plt.plot(part["availability"], part["mean_ndcg"], marker="o", label=f"{n} components")
plt.xlabel("Training data availability (%)")
plt.ylabel("Mean NDCG@10")
plt.title("Collaborative filtering: SVD components vs. data availability")
plt.xticks([20, 40, 60, 80, 100])
plt.legend(title="SVD dimensionality")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(out / "svd_component_comparison.png", dpi=180)
plt.close()

print("\nMean NDCG@10 by availability and SVD components:")
print(summary.pivot(index="availability", columns="n_components", values="mean_ndcg").round(6).to_string())
print(f"\nSaved {len(df)} evaluations, summary CSV, and comparison graph.")
