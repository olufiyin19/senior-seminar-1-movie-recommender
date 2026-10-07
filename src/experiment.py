import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def ndcg_at_k(recommended_movie_ids, relevant_movie_ids, k=10):
    recommended_movie_ids = recommended_movie_ids[:k]

    dcg = 0.0

    for rank, movie_id in enumerate(recommended_movie_ids, start=1):
        if movie_id in relevant_movie_ids:
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_movie_ids), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg

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

def evaluate_collaborative(
    reduced_train_ratings,
    test_ratings,
    fixed_user_ids,
    fixed_movie_ids,
    original_train_ratings,
    k=10
):

    # Build the user-movie matrix from this training condition
    user_movie_matrix = reduced_train_ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0
    )

    # Train TruncatedSVD
    svd = TruncatedSVD(
        n_components=50,
        random_state=42
    )

    user_factors = svd.fit_transform(user_movie_matrix)
    predicted_ratings = svd.inverse_transform(user_factors)
        # Find movies available in this training condition
   # Use the same movie catalog across all experiments
    training_movie_ids = set(fixed_movie_ids)

    # Keep only test ratings from the fixed movie catalog
    eligible_test_ratings = test_ratings[
        test_ratings["movieId"].isin(fixed_movie_ids)
    ]

    # Relevant items are movies rated 4 or higher
    relevant_test_ratings = eligible_test_ratings[
        eligible_test_ratings["rating"] >= 4.0
    ]

    # Use the same evaluation users across all experiments
    eligible_users = set(fixed_user_ids)

    # Calculate NDCG@K for each eligible user
    ndcg_scores = []

    for user_id in sorted(eligible_users):

        # Find the user's row in the matrix
        user_index = user_movie_matrix.index.get_loc(user_id)

        # Get predicted scores for this user
        user_scores = predicted_ratings[user_index]

        # Get movies already seen in this training condition
        # Exclude all movies rated in the original training set
        seen_movies = set(
            original_train_ratings.loc[
            original_train_ratings["userId"] == user_id,
            "movieId"
        ]
    )

        # Create recommendation candidates from the fixed movie catalog
        recommendations = pd.DataFrame({
            "movieId": user_movie_matrix.columns,
            "predicted_score": user_scores
        })

        # Keep only movies available in every experiment
        recommendations = recommendations[
            recommendations["movieId"].isin(fixed_movie_ids)
        ]

        # Remove movies already seen during training
        recommendations = recommendations[
            ~recommendations["movieId"].isin(seen_movies)
        ]

        # Select Top-K recommendations
        top_k_movie_ids = (
            recommendations
            .sort_values("predicted_score", ascending=False)
            .head(k)["movieId"]
            .tolist()
        )

        # Get this user's relevant test movies
        relevant_movies = set(
            relevant_test_ratings.loc[
                relevant_test_ratings["userId"] == user_id,
                "movieId"
            ]
        )

        # Calculate this user's NDCG
        user_ndcg = ndcg_at_k(
            top_k_movie_ids,
            relevant_movies,
            k=k
        )

        ndcg_scores.append(user_ndcg)

    # Return the aggregate result
    return {
        "users_evaluated": len(ndcg_scores),
        "mean_ndcg": np.mean(ndcg_scores)
    }

# Load MovieLens ratings data
ratings = pd.read_csv("data/ml-latest-small-master/ratings.csv")

print("Total ratings:", len(ratings))
print("Total users:", ratings["userId"].nunique())
print("Total movies:", ratings["movieId"].nunique())

# Create one fixed 80/20 train-test split for each user
train_parts = []
test_parts = []

for user_id, user_ratings in ratings.groupby("userId"):
    user_ratings = user_ratings.sample(
        frac=1,
        random_state=42
    )

    split_index = int(len(user_ratings) * 0.8)

    train_parts.append(user_ratings.iloc[:split_index])
    test_parts.append(user_ratings.iloc[split_index:])

train_ratings = pd.concat(train_parts, ignore_index=True)
test_ratings = pd.concat(test_parts, ignore_index=True)

print("\nFixed train/test split:")
print("Training ratings:", len(train_ratings))
print("Test ratings:", len(test_ratings))
print("Combined:", len(train_ratings) + len(test_ratings))

# Verify every user appears in both sets
print("\nUsers in each split:")
print("Training users:", train_ratings["userId"].nunique())
print("Test users:", test_ratings["userId"].nunique())

# Test each training-data availability level
availability_levels = [0.20, 0.40, 0.60, 0.80, 1.00]
subsample_seeds = [42, 43, 44, 45, 46]

# Generate and store all training conditions
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

# Find movies available in every training condition
common_movie_ids = set(train_ratings["movieId"].unique())

for condition_ratings in training_conditions.values():
    common_movie_ids &= set(condition_ratings["movieId"].unique())

print("Movies available in all conditions:", len(common_movie_ids))

# Identify users eligible in every training condition
common_user_ids = set(train_ratings["userId"].unique())

for condition_ratings in training_conditions.values():
    ratings_per_user = condition_ratings.groupby("userId").size()

    eligible_user_ids = set(
        ratings_per_user[ratings_per_user >= 5].index
    )

    common_user_ids &= eligible_user_ids

# Keep fixed test ratings for movies available in all conditions
common_test_ratings = test_ratings[
    test_ratings["movieId"].isin(common_movie_ids)
].copy()

# Identify users with at least one relevant common test movie
users_with_relevant_test_movies = set(
    common_test_ratings.loc[
        common_test_ratings["rating"] >= 4.0,
        "userId"
    ]
)

common_user_ids &= users_with_relevant_test_movies

print("Users eligible in all conditions:", len(common_user_ids))
print("Common test ratings:", len(common_test_ratings))

# Keep only test ratings belonging to our fixed evaluation users
common_test_ratings = common_test_ratings[
    common_test_ratings["userId"].isin(common_user_ids)
].copy()

print("Fixed evaluation users:", len(common_user_ids))
print("Fixed evaluation test ratings:", len(common_test_ratings))

for availability in availability_levels:
    # Reduced-data conditions use five seeds.
    # 100% uses the full training set, so only one run is needed.
    seeds_to_run = subsample_seeds if availability < 1.00 else [42]
    for subsample_seed in seeds_to_run:

        if availability == 1.00:
            reduced_train_ratings = train_ratings.copy()
        else:
            reduced_parts = []

            for user_id, user_ratings in train_ratings.groupby("userId"):
                reduced_user_ratings = user_ratings.sample(
                    frac=availability,
                    random_state=subsample_seed
                )

                reduced_parts.append(reduced_user_ratings)

            reduced_train_ratings = pd.concat(
                reduced_parts,
                ignore_index=True
            )

        # Check user eligibility
        ratings_per_user = reduced_train_ratings.groupby("userId").size()

        eligible_users = ratings_per_user[
            ratings_per_user >= 5
        ]

        print(
            f"\n{int(availability * 100)}% training availability "
            f"(seed {subsample_seed}):"
        )
        print("Ratings retained:", len(reduced_train_ratings))
        print("Eligible users:", len(eligible_users))
        print("Ineligible users:", 610 - len(eligible_users))
        print("Minimum ratings retained:", ratings_per_user.min())


# Check movie coverage for the 20% availability condition, seed 42
availability = 0.20
coverage_seed = 42

coverage_parts = []

for user_id, user_ratings in train_ratings.groupby("userId"):
    reduced_user_ratings = user_ratings.sample(
        frac=availability,
        random_state=coverage_seed
    )
    coverage_parts.append(reduced_user_ratings)

coverage_train_ratings = pd.concat(
    coverage_parts,
    ignore_index=True
)

# Movies that appear in this reduced training set
training_movie_ids = set(coverage_train_ratings["movieId"])

print("\n20% seed 42 movie coverage:")
print("Movies in reduced training set:", len(training_movie_ids))

# Check which test ratings refer to movies seen during training
eligible_test_ratings = test_ratings[
    test_ratings["movieId"].isin(training_movie_ids)
]

ineligible_test_ratings = test_ratings[
    ~test_ratings["movieId"].isin(training_movie_ids)
]

print("\nTest item coverage:")
print("Total fixed test ratings:", len(test_ratings))
print("Test ratings with movies seen in training:", len(eligible_test_ratings))
print("Test ratings with unseen movies:", len(ineligible_test_ratings))

# Keep only relevant test ratings
relevant_test_ratings = eligible_test_ratings[
    eligible_test_ratings["rating"] >= 4.0
]

print("\nRelevant test items:")
print("Eligible test ratings:", len(eligible_test_ratings))
print("Relevant test ratings (rating >= 4):", len(relevant_test_ratings))
print(
    "Users with at least one relevant test item:",
    relevant_test_ratings["userId"].nunique()
)

# Combine user eligibility requirements
training_eligible_user_ids = set(
    coverage_train_ratings.groupby("userId")
    .size()
    .loc[lambda x: x >= 5]
    .index
)

relevant_test_user_ids = set(
    relevant_test_ratings["userId"].unique()
)

final_eligible_users = (
    training_eligible_user_ids &
    relevant_test_user_ids
)

print("\nFinal evaluation eligibility:")
print("Training-eligible users:", len(training_eligible_user_ids))
print("Users with relevant test items:", len(relevant_test_user_ids))
print("Users meeting both requirements:", len(final_eligible_users))

# Build collaborative model for 20% availability, seed 42
experiment_user_movie_matrix = coverage_train_ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0
)

print("\n20% experimental user-movie matrix:")
print("Shape:", experiment_user_movie_matrix.shape)

# Train TruncatedSVD on the reduced training matrix
experiment_svd = TruncatedSVD(
    n_components=50,
    random_state=42
)

experiment_user_factors = experiment_svd.fit_transform(
    experiment_user_movie_matrix
)

experiment_predicted_ratings = experiment_svd.inverse_transform(
    experiment_user_factors
)

print("\n20% experimental SVD:")
print("User factors shape:", experiment_user_factors.shape)
print(
    "Predicted ratings shape:",
    experiment_predicted_ratings.shape
)

# Select one eligible user for a test evaluation
sample_user = sorted(final_eligible_users)[0]

print("\nSample evaluation user:", sample_user)

# Find the user's row in the experimental matrix
sample_user_index = experiment_user_movie_matrix.index.get_loc(
    sample_user
)

# Get predicted collaborative scores
sample_user_scores = experiment_predicted_ratings[
    sample_user_index
]

# Movies this user has already seen in the reduced training set
sample_user_seen_movies = set(
    coverage_train_ratings.loc[
        coverage_train_ratings["userId"] == sample_user,
        "movieId"
    ]
)

# Pair movie IDs with predicted scores
sample_recommendations = pd.DataFrame({
    "movieId": experiment_user_movie_matrix.columns,
    "predicted_score": sample_user_scores
})

# Do not recommend movies already seen during training
sample_recommendations = sample_recommendations[
    ~sample_recommendations["movieId"].isin(
        sample_user_seen_movies
    )
]

# Select the 10 highest-scoring unseen movies
sample_recommendations = sample_recommendations.sort_values(
    "predicted_score",
    ascending=False
).head(10)

print("\nUser 1 Top-10 collaborative movie IDs:")
print(sample_recommendations["movieId"].tolist())

# Get this user's relevant movies from the eligible fixed test set
sample_relevant_movies = set(
    relevant_test_ratings.loc[
        relevant_test_ratings["userId"] == sample_user,
        "movieId"
    ]
)

# Get the recommended movie IDs
sample_recommended_movie_ids = (
    sample_recommendations["movieId"].tolist()
)

# Calculate NDCG@10
sample_ndcg = ndcg_at_k(
    sample_recommended_movie_ids,
    sample_relevant_movies,
    k=10
)

print("\nUser 1 evaluation:")
print("Relevant test movie IDs:", sorted(sample_relevant_movies))
print("Recommended movie IDs:", sample_recommended_movie_ids)
print("NDCG@10:", sample_ndcg)

# Evaluate collaborative NDCG@10 across all eligible users
collaborative_ndcg_scores = []

for user_id in sorted(final_eligible_users):

    # Find user's row in the experimental matrix
    user_index = experiment_user_movie_matrix.index.get_loc(
        user_id
    )

    # Get predicted scores
    user_scores = experiment_predicted_ratings[user_index]

    # Get movies already seen in reduced training data
    seen_movies = set(
        coverage_train_ratings.loc[
            coverage_train_ratings["userId"] == user_id,
            "movieId"
        ]
    )

    # Create recommendation candidates
    user_recommendations = pd.DataFrame({
        "movieId": experiment_user_movie_matrix.columns,
        "predicted_score": user_scores
    })

    # Remove already-seen movies
    user_recommendations = user_recommendations[
        ~user_recommendations["movieId"].isin(seen_movies)
    ]

    # Select Top 10
    top_10_movie_ids = (
        user_recommendations
        .sort_values("predicted_score", ascending=False)
        .head(10)["movieId"]
        .tolist()
    )

    # Get relevant fixed-test movies
    relevant_movies = set(
        relevant_test_ratings.loc[
            relevant_test_ratings["userId"] == user_id,
            "movieId"
        ]
    )

    # Calculate this user's NDCG@10
    user_ndcg = ndcg_at_k(
        top_10_movie_ids,
        relevant_movies,
        k=10
    )

    collaborative_ndcg_scores.append(user_ndcg)

# Average across eligible users
mean_collaborative_ndcg = np.mean(
    collaborative_ndcg_scores
)

print("\n20% seed 42 collaborative evaluation:")
print("Users evaluated:", len(collaborative_ndcg_scores))
print("Mean NDCG@10:", mean_collaborative_ndcg)

# Store results from all collaborative experiments
collaborative_results = []
# Run collaborative evaluation across all availability levels
for availability in availability_levels:

    # Reduced-data conditions use five seeds.
    # 100% uses the full training set once.
    seeds_to_run = subsample_seeds if availability < 1.00 else [42]

    for seed in seeds_to_run:

        # Create the training set for this condition
        if availability == 1.00:
            current_train_ratings = train_ratings.copy()
        else:
            current_parts = []

            for user_id, user_ratings in train_ratings.groupby("userId"):
                sampled_ratings = user_ratings.sample(
                    frac=availability,
                    random_state=seed
                )
                current_parts.append(sampled_ratings)

            current_train_ratings = pd.concat(
                current_parts,
                ignore_index=True
            )

        print(
            f"\nRunning collaborative evaluation: "
            f"{int(availability * 100)}%, seed {seed}"
        )

        # Evaluate this condition
        # Evaluate using the same users and movies in every condition
        result = evaluate_collaborative(
            current_train_ratings,
            common_test_ratings,
            common_user_ids,
            common_movie_ids,
            train_ratings
        )

        # Save the result
        collaborative_results.append({
            "availability": int(availability * 100),
            "seed": seed,
            "users_evaluated": result["users_evaluated"],
            "mean_ndcg": result["mean_ndcg"]
        })

        print("Users evaluated:", result["users_evaluated"])
        print("Mean NDCG@10:", result["mean_ndcg"])

# Convert results to a DataFrame
collaborative_results_df = pd.DataFrame(collaborative_results)

# Save results to CSV
collaborative_results_df.to_csv(
    "results/collaborative_midterm_results.csv",
    index=False
)

print("\nAll collaborative experiment results:")
print(collaborative_results_df)

print(
    "\nResults saved to "
    "results/collaborative_midterm_results.csv"
)

# Summarize mean performance at each availability level
collaborative_summary = (
    collaborative_results_df
    .groupby("availability")
    .agg(
        average_ndcg=("mean_ndcg", "mean"),
        std_ndcg=("mean_ndcg", "std"),
        average_users=("users_evaluated", "mean")
    )
    .reset_index()
)

print("\nCollaborative summary by training availability:")
print(collaborative_summary)