"""
validate_against_ground_truth.py
The composite score and any model built from it use ONLY behavioral
features -- persona is never an input. This script is the honesty check:
does the resulting score actually separate personas correctly, even though
it was never told what a persona is?

Two independent validations are run:
  1. Composite score by persona (should rank: calm < average < aggressive,
     and drowsy_distracted should score high due to inconsistent/erratic
     behavior even though it isn't the fastest persona).
  2. Unsupervised KMeans clustering on the same features, compared against
     persona via a contingency table and Adjusted Rand Index -- this
     validates the FEATURES themselves separate driving styles, independent
     of any scoring formula or weighting choice.
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from feature_engineering import build_trip_features
from risk_scoring import compute_composite_risk_score, assign_risk_tier


def validate_composite_score(trips: pd.DataFrame, features: pd.DataFrame, scores: pd.Series, out_dir: str):
    df = trips.set_index("trip_id").copy()
    df["risk_score"] = scores

    order = ["calm", "average", "aggressive", "drowsy_distracted"]
    print("Composite risk score by persona (should increase left to right,")
    print("with drowsy_distracted elevated due to inconsistency, not raw speed):")
    summary = df.groupby("persona")["risk_score"].agg(["mean", "median", "std"]).reindex(order)
    print(summary.round(2))

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["persona"] == p]["risk_score"] for p in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title("Composite Risk Score by Persona\n(persona is NOT a model input -- this is post-hoc validation)")
    ax.set_ylabel("Risk score (0-100)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/risk_score_by_persona.png", dpi=120)
    plt.close()

    return summary


def validate_with_clustering(trips: pd.DataFrame, features: pd.DataFrame, out_dir: str, n_clusters: int = 4):
    df = trips.set_index("trip_id")
    persona = df.loc[features.index, "persona"]

    X = StandardScaler().fit_transform(features.values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)

    ari = adjusted_rand_score(persona, cluster_labels)
    print(f"\nUnsupervised KMeans (k={n_clusters}) vs persona ground truth:")
    print(f"Adjusted Rand Index: {ari:.3f}  "
          f"(0 = random agreement, 1 = perfect agreement; "
          f"note ARI penalizes cluster-count mismatch, so this is a lower bound)")

    contingency = pd.crosstab(persona, cluster_labels, normalize="index")
    print("\nContingency table (row-normalized -- what fraction of each persona")
    print("landed in each unsupervised cluster):")
    print(contingency.round(2))

    fig, ax = plt.subplots(figsize=(7, 5))
    contingency.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title(f"Unsupervised Cluster Assignment by True Persona (k={n_clusters})")
    ax.set_ylabel("Fraction of trips")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cluster_vs_persona.png", dpi=120)
    plt.close()

    return ari, contingency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--out", type=str, default="./eda_output")
    args = parser.parse_args()

    trips = pd.read_csv(args.data)
    features = build_trip_features(trips)
    scores = compute_composite_risk_score(features)

    validate_composite_score(trips, features, scores, args.out)
    validate_with_clustering(trips, features, args.out)
