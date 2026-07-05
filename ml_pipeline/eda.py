"""
eda.py
Exploratory analysis of trips.csv. Produces plots that visually confirm the
patterns we expect to exist (persona -> behavior, vehicle age -> degradation)
before any modeling begins. This is a validation step, not decoration --
if these plots didn't show the expected patterns, the simulator or feature
engineering would need fixing before trusting any model built on top of it.

Usage:
    python eda.py --data ./data/trips.csv --out ./eda_output
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_trips(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["trip_start_time"])
    return df


def plot_persona_behavior(df: pd.DataFrame, out_dir: str):
    """Confirms harsh events and speeding scale with persona aggressiveness."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    order = ["calm", "average", "aggressive", "drowsy_distracted"]

    metrics = [
        ("harsh_braking_count", "Harsh Braking Events per Trip"),
        ("harsh_accel_count", "Harsh Acceleration Events per Trip"),
        ("overspeeding_events", "Overspeeding Events per Trip"),
        ("harsh_cornering_count", "Harsh Cornering Events per Trip"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics):
        data = [df[df["persona"] == p][col] for p in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "persona_behavior_boxplots.png"), dpi=120)
    plt.close()


def plot_speed_ratio_by_persona(df: pd.DataFrame, out_dir: str):
    df = df.copy()
    df["speed_ratio"] = df["avg_speed_kmh"] / df["speed_limit_kmh"]
    order = ["calm", "average", "aggressive", "drowsy_distracted"]

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["persona"] == p]["speed_ratio"] for p in order]
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=20)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Speed limit")
    ax.set_title("Avg Speed / Speed Limit Ratio by Persona")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed_ratio_by_persona.png"), dpi=120)
    plt.close()


def plot_degradation_curves(df: pd.DataFrame, out_dir: str):
    """
    Confirms engine temp / brake efficiency actually drift with vehicle age
    for vehicles with an assigned failure mode, vs. staying flat for
    healthy_control vehicles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for mode in df["vehicle_failure_mode"].unique():
        sub = df[df["vehicle_failure_mode"] == mode]
        if mode == "engine_overheat":
            binned = sub.groupby(sub["vehicle_age_days"] // 10 * 10)["avg_engine_temp_c"].mean()
            axes[0].plot(binned.index, binned.values, label=mode)
        elif mode == "healthy_control":
            binned = sub.groupby(sub["vehicle_age_days"] // 10 * 10)["avg_engine_temp_c"].mean()
            axes[0].plot(binned.index, binned.values, label=mode, linestyle="--")

    axes[0].set_title("Avg Engine Temp vs Vehicle Age\n(engine_overheat vs healthy_control)")
    axes[0].set_xlabel("Vehicle age (days)")
    axes[0].set_ylabel("Engine temp (C)")
    axes[0].legend()

    for mode in ["brake_wear", "healthy_control"]:
        sub = df[df["vehicle_failure_mode"] == mode]
        binned = sub.groupby(sub["vehicle_age_days"] // 10 * 10)["brake_efficiency_pct"].mean()
        axes[1].plot(binned.index, binned.values, label=mode)

    axes[1].set_title("Brake Efficiency vs Vehicle Age\n(brake_wear vs healthy_control)")
    axes[1].set_xlabel("Vehicle age (days)")
    axes[1].set_ylabel("Brake efficiency (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "degradation_curves.png"), dpi=120)
    plt.close()


def plot_maintenance_label_balance(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["will_fail_within_30d"].value_counts().sort_index()
    ax.bar(["No (0)", "Yes (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_title("will_fail_within_30d Label Balance (trip-level)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}\n({v/counts.sum()*100:.1f}%)", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "maintenance_label_balance.png"), dpi=120)
    plt.close()


def print_summary(df: pd.DataFrame):
    print("=" * 70)
    print("TRIPS SUMMARY")
    print("=" * 70)
    print(f"Total trips: {len(df):,}")
    print(f"Date range: {df['trip_start_time'].min()} -> {df['trip_start_time'].max()}")
    print(f"Unique drivers: {df['driver_id'].nunique()}, Unique vehicles: {df['vehicle_id'].nunique()}")
    print()
    print("Persona distribution:")
    print(df["persona"].value_counts(normalize=True).round(3))
    print()
    print("Correlation: harsh_braking_count vs persona (mean per persona):")
    print(df.groupby("persona")["harsh_braking_count"].mean().round(2))
    print()
    print("will_fail_within_30d positive rate:", round(df["will_fail_within_30d"].mean() * 100, 2), "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/trips.csv")
    parser.add_argument("--out", type=str, default="./eda_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trips = load_trips(args.data)

    print_summary(trips)
    plot_persona_behavior(trips, args.out)
    plot_speed_ratio_by_persona(trips, args.out)
    plot_degradation_curves(trips, args.out)
    plot_maintenance_label_balance(trips, args.out)

    print(f"\nPlots saved to {args.out}/")
