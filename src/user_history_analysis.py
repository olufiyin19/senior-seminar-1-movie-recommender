
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")

# Find the individual-user results from all training conditions.
files = sorted(RESULTS.glob("user_history_*_seed_*.csv"))

if len(files) != 21:
    raise RuntimeError(
        f"Expected 21 user-history CSV files, found {len(files)}. "
        "Finish the full experiment before running this analysis."
    )

df = pd.concat(
    [pd.read_csv(path) for path in files],
    ignore_index=True
)

# Categorize users based on the number of ratings
# actually available to the trained model.
def history_group(count):
    if count < 20:
        return "Sparse (1-19)"
    elif count < 100:
        return "Moderate (20-99)"
    return "Extensive (100+)"


df["history_group"] = df["training_history_count"].apply(
    history_group
)

# Check that selected and baseline scores are available
# for the same users in each training condition.
keys = ["availability", "seed", "userId"]

paired = df.pivot(
    index=keys + ["history_group", "training_history_count"],
    columns="method",
    values="ndcg"
).reset_index()

required = ["validation_selected", "fixed_collaborative"]

if paired[required].isna().any().any():
    raise RuntimeError(
        "Some users are missing a selected or baseline score."
    )

paired["delta_ndcg"] = (
    paired["validation_selected"]
    - paired["fixed_collaborative"]
)

# Save one row per user and training condition.
paired.to_csv(
    RESULTS / "user_history_paired.csv",
    index=False
)

# Aggregate within each condition first, rather than
# treating repeated appearances of users as independent.
condition_summary = (
    paired.groupby(
        ["availability", "seed", "history_group"],
        as_index=False
    )
    .agg(
        users=("userId", "nunique"),
        selected_ndcg=("validation_selected", "mean"),
        baseline_ndcg=("fixed_collaborative", "mean"),
        delta_ndcg=("delta_ndcg", "mean"),
    )
)

condition_summary.to_csv(
    RESULTS / "user_history_condition_summary.csv",
    index=False
)

# Summarize across subsampling seeds.
summary = (
    condition_summary.groupby(
        ["availability", "history_group"],
        as_index=False
    )
    .agg(
        conditions=("seed", "size"),
        mean_users=("users", "mean"),
        selected_ndcg=("selected_ndcg", "mean"),
        baseline_ndcg=("baseline_ndcg", "mean"),
        mean_delta=("delta_ndcg", "mean"),
        std_delta=("delta_ndcg", "std"),
    )
)

summary.to_csv(
    RESULTS / "user_history_summary.csv",
    index=False
)

# Plot the difference between selected and baseline models.
fig, ax = plt.subplots(figsize=(9, 5.5))

group_order = [
    "Sparse (1-19)",
    "Moderate (20-99)",
    "Extensive (100+)"
]

for group in group_order:
    subset = summary[
        summary["history_group"] == group
    ].sort_values("availability")

    if not subset.empty:
        ax.plot(
            subset["availability"],
            subset["mean_delta"],
            marker="o",
            label=group
        )

ax.axhline(0, color="black", linestyle="--", linewidth=1)

ax.set(
    xlabel="Training data availability (%)",
    ylabel="Selected minus baseline NDCG@10",
    title="Recommendation improvement by user rating history"
)

ax.set_xticks([20, 40, 60, 80, 100])
ax.legend(title="Rating history")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    RESULTS / "user_history_comparison.png",
    dpi=180
)
plt.close(fig)

print("\nUser-history analysis summary:")
print(
    summary.round(6).to_string(index=False)
)

print("\nSaved:")
print("  results/user_history_paired.csv")
print("  results/user_history_condition_summary.csv")
print("  results/user_history_summary.csv")
print("  results/user_history_comparison.png")
