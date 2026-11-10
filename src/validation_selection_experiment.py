
""" validation-selected SVD dimensionality and hybrid weight.
Pilot: seed 42 at five availability levels. No final-test labels used for tuning.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA = Path("data/ml-latest-small-master")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

COMPONENTS = [10, 25, 50, 75, 100]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
AVAILABILITIES = [0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [42, 43, 44, 45, 46]
K = 10

ratings = pd.read_csv(DATA / "ratings.csv")
movies = pd.read_csv(DATA / "movies.csv")


def split_by_user(frame, train_fraction, seed):
    """Split each user's ratings into two disjoint groups."""
    a, b = [], []

    for _, part in frame.groupby("userId", sort=True):
        shuffled = part.sample(frac=1, random_state=seed)
        cut = int(len(shuffled) * train_fraction)

        a.append(shuffled.iloc[:cut])
        b.append(shuffled.iloc[cut:])

    return (
        pd.concat(a, ignore_index=True),
        pd.concat(b, ignore_index=True),
    )


def subsample(frame, availability, seed):
    """Keep a specified fraction of each user's training ratings."""
    if availability == 1.0:
        return frame.copy()

    return pd.concat(
        [
            part.sample(frac=availability, random_state=seed)
            for _, part in frame.groupby("userId", sort=True)
        ],
        ignore_index=True,
    )


def normalize(v):
    """Scale recommendation scores to the range [0, 1]."""
    v = np.asarray(v, dtype=float)

    if np.isclose(v.min(), v.max()):
        return np.zeros_like(v)

    return (v - v.min()) / (v.max() - v.min())


def ndcg(top_ids, relevant):
    """Calculate NDCG@10 using binary relevance."""
    if not relevant:
        return np.nan

    gain = np.array(
        [movie in relevant for movie in top_ids[:K]],
        dtype=float,
    )

    dcg = np.sum(
        gain / np.log2(np.arange(2, len(gain) + 2))
    )

    ideal = min(len(relevant), K)

    idcg = np.sum(
        1 / np.log2(np.arange(2, ideal + 2))
    )

    return float(dcg / idcg)


# Build genre features from movie metadata.
# These do not depend on held-out user ratings.
vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")

genre_matrix = vectorizer.fit_transform(
    movies["genres"].fillna("")
)

movie_to_idx = {
    mid: i
    for i, mid in enumerate(movies["movieId"])
}


def score_grid(fit_ratings, eval_ratings, exclude_ratings, return_user_scores=False):
    """
    Evaluate every combination of SVD components and hybrid weights.

    Models are trained only on fit_ratings.
    Evaluation ratings are used to calculate NDCG, not fit models.
    """

    # Create the user-movie rating matrix.
    matrix = fit_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0,
    )

    # Candidate movies must appear in the training matrix.
    candidates = np.array(
        sorted(set(matrix.columns) & set(movie_to_idx))
    )

    columns = matrix.columns.get_indexer(candidates)

    genre_rows = np.array(
        [movie_to_idx[mid] for mid in candidates]
    )

    user_row = {
        uid: i
        for i, uid in enumerate(matrix.index)
    }

    # Movies that must not be recommended to each user.
    seen = (
        exclude_ratings.groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    # Movies rated at least four stars in the fitting data.
    liked = (
        fit_ratings[fit_ratings["rating"] >= 4]
        .groupby("userId")["movieId"]
        .agg(list)
        .to_dict()
    )

    # Relevant evaluation movies.
    relevant = (
        eval_ratings[
            (eval_ratings["rating"] >= 4)
            & eval_ratings["movieId"].isin(candidates)
        ]
        .groupby("userId")["movieId"]
        .agg(set)
        .to_dict()
    )

    users = sorted(set(user_row) & set(relevant))

    if not users:
        raise RuntimeError(
            "No users with relevant evaluation movies in candidate catalog"
        )

    # Cache content-based scores because they do not change
    # when we change the number of SVD components.
    cache = {}

    for uid in users:
        idx = [
            movie_to_idx[mid]
            for mid in liked.get(uid, [])
            if mid in movie_to_idx
        ]

        if idx:
            profile = np.asarray(
                genre_matrix[idx].mean(axis=0)
            )

            content = cosine_similarity(
                profile,
                genre_matrix[genre_rows],
            ).ravel()

        else:
            content = np.zeros(len(candidates))

        cache[uid] = (
            normalize(content),
            ~np.isin(
                candidates,
                list(seen.get(uid, set())),
            ),
        )

    scores = []
    user_scores = []

    # Test each SVD component count.
    for n in COMPONENTS:

        if n >= min(matrix.shape):
            raise ValueError(
                f"{n} SVD components exceed matrix limit {matrix.shape}"
            )

        svd = TruncatedSVD(
            n_components=n,
            random_state=42,
        )

        reconstructed = svd.inverse_transform(
            svd.fit_transform(matrix)
        )

        per_weight = {
            w: []
            for w in WEIGHTS
        }

        # Evaluate every eligible user.
        for uid in users:
            content_norm, unseen = cache[uid]

            collab_norm = normalize(
                reconstructed[user_row[uid], columns]
            )

            eligible_movies = candidates[unseen]

            # Try each hybrid weight.
            for w in WEIGHTS:
                combined = (
                    w * collab_norm
                    + (1 - w) * content_norm
                )[unseen]

                top = eligible_movies[
                    np.argsort(
                        -combined,
                        kind="stable",
                    )[:K]
                ]

                user_ndcg = ndcg(top, relevant[uid])
                per_weight[w].append(user_ndcg)

                if return_user_scores:
                    user_scores.append({
                        "userId": uid,
                        "n_components": n,
                        "collaborative_weight": w,
                        "training_history_count": int(
                            (fit_ratings["userId"] == uid).sum()
                        ),
                        "ndcg": user_ndcg,
                    })

        # Store average scores for each weight.
        for w in WEIGHTS:
            scores.append(
                {
                    "n_components": n,
                    "collaborative_weight": w,
                    "mean_ndcg": float(
                        np.mean(per_weight[w])
                    ),
                    "users_evaluated": len(users),
                    "candidate_movies": len(candidates),
                }
            )
    
    if return_user_scores:
        return pd.DataFrame(scores), pd.DataFrame(user_scores)

    return pd.DataFrame(scores)


# STEP 1: CREATE TRAINING, VALIDATION, AND TEST DATA

# Preserve the original outer 80/20 train/test split.
outer_train, outer_test = split_by_user(
    ratings,
    0.8,
    42,
)

# Split the original training data again.
# The final test ratings are not involved.
inner_train, inner_validation = split_by_user(
    outer_train,
    0.8,
    24,
)


# STEP 2: VALIDATION-BASED MODEL SELECTION

validation_rows = []
test_rows = []

for availability in AVAILABILITIES:

    for seed in (SEEDS if availability < 1 else [42]):

        print(
            f"Availability {availability:.0%}, seed {seed}: "
            "validation grid",
            flush=True,
        )

        # Reduce the training data to the chosen availability.
        tuning_train = subsample(
            inner_train,
            availability,
            seed,
        )

        # Evaluate all component/weight combinations
        # using validation ratings.
        validation = score_grid(
            tuning_train,
            inner_validation,
            tuning_train,
        )

        validation.insert(0, "seed", seed)
        validation.insert(
            0,
            "availability",
            int(availability * 100),
        )

        validation_rows.append(validation)

        # Choose the highest validation NDCG.
        # Break ties using fewer components, then lower weight.
        chosen = validation.sort_values(
            [
                "mean_ndcg",
                "n_components",
                "collaborative_weight",
            ],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]

        n = int(chosen["n_components"])
        w = float(chosen["collaborative_weight"])

        print(
            f"  Selected {n} components, weight {w:.2f}; "
            f"validation NDCG@10={chosen.mean_ndcg:.6f}",
            flush=True,
        )

      
        # STEP 3: FINAL HELD-OUT TEST EVALUATION

        # Refit using the available portion of outer training.
        final_train = subsample(
            outer_train,
            availability,
            seed,
        )

        # Exclude movies rated in the original outer training
        # history, matching the earlier experiments.
        test_grid, individual_test_scores = score_grid(
            final_train,
            outer_test,
            outer_train,
            return_user_scores=True,
        )

        individual_test_scores["availability"] = int(availability * 100)
        individual_test_scores["seed"] = seed

        selected_mask = (
            (individual_test_scores["n_components"] == n)
            & (individual_test_scores["collaborative_weight"] == w)
        )

        baseline_mask = (
            (individual_test_scores["n_components"] == 50)
            & (individual_test_scores["collaborative_weight"] == 1.0)
        )

        selected_users = individual_test_scores[selected_mask].copy()
        selected_users["method"] = "validation_selected"

        baseline_users = individual_test_scores[baseline_mask].copy()
        baseline_users["method"] = "fixed_collaborative"

        user_output = pd.concat(
            [selected_users, baseline_users],
            ignore_index=True,
        )

        user_output.to_csv(
            OUT / f"user_history_{int(availability * 100)}_seed_{seed}.csv",
            index=False,
        )

        # Find the test result of the validation-selected model.
        selected = test_grid[
            (test_grid["n_components"] == n)
            & (test_grid["collaborative_weight"] == w)
        ].iloc[0]

        # Fixed collaborative reference: 50 SVD components.
        reference = test_grid[
            (test_grid["n_components"] == 50)
            & (test_grid["collaborative_weight"] == 1.0)
        ].iloc[0]

        test_rows.append(
            {
                "availability": int(availability * 100),
                "seed": seed,
                "selected_components": n,
                "selected_collaborative_weight": w,
                "validation_ndcg": float(
                    chosen["mean_ndcg"]
                ),
                "selected_test_ndcg": float(
                    selected["mean_ndcg"]
                ),
                "reference_test_ndcg": float(
                    reference["mean_ndcg"]
                ),
                "test_delta_ndcg": float(
                    selected["mean_ndcg"]
                    - reference["mean_ndcg"]
                ),
                "validation_users": int(
                    chosen["users_evaluated"]
                ),
                "test_users": int(
                    selected["users_evaluated"]
                ),
                "validation_candidates": int(
                    chosen["candidate_movies"]
                ),
                "test_candidates": int(
                    selected["candidate_movies"]
                ),
            }
        )

        print(
            f"  Test: selected={selected.mean_ndcg:.6f}, "
            f"reference={reference.mean_ndcg:.6f}",
            flush=True,
        )


# STEP 4: SAVE RESULTS

validation_df = pd.concat(
    validation_rows,
    ignore_index=True,
)

test_df = pd.DataFrame(test_rows)

validation_df.to_csv(
    OUT / "validation_selection_grid.csv",
    index=False,
)

test_df.to_csv(
    OUT / "validation_selection_results.csv",
    index=False,
)


# STEP 5: GENERATE COMPARISON GRAPH

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    test_df["availability"],
    test_df["selected_test_ndcg"],
    marker="o",
    label="Validation-selected model",
)

ax.plot(
    test_df["availability"],
    test_df["reference_test_ndcg"],
    marker="s",
    label="Fixed 50-component collaborative reference",
)

ax.set(
    xlabel="Training data availability (%)",
    ylabel="Test NDCG@10",
    title=(
        "Validation-selected recommendation model "
        "vs fixed reference"
    ),
)

ax.set_xticks([20, 40, 60, 80, 100])

ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUT / "validation_selection_comparison.png",
    dpi=180,
)

plt.close(fig)


# STEP 6: DISPLAY FINAL RESULTS

print(
    "\nValidation-selected configurations "
    "and held-out test results:"
)

print(test_df.to_string(index=False))

print(
    "\nSaved validation_selection_grid.csv, "
    "validation_selection_results.csv, "
    "validation_selection_comparison.png"
)

print(
    "NOTE: Validation and test candidate catalogs may differ; "
    "compare models within each test condition."
)
