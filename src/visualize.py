"""
matplotlib plotting
"""

import matplotlib.pyplot as plt
import numpy as np

TEAM_COLORS = {
    "RED BULL": "#0600EF", "FERRARI": "#DC0000", "MERCEDES": "#00D2BE",
    "MCLAREN": "#FF8700", "ASTON MARTIN": "#006F62", "ALPINE": "#0090FF",
    "WILLIAMS": "#005AFF", "ALPHATAURI": "#2B4562", "ALFA ROMEO": "#900000",
    "HAAS": "#FFFFFF",
}

COMPOUND_COLORS = {"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC",
                   "INTERMEDIATE": "#43B02A", "WET": "#0067AD"}

def plot_stint_strategy(stint_df, title="Tire Strat by Driver", save_path=None):
    """Horizontal timeline"""
    drivers = stint_df["Driver"].unique()
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(drivers) + 1.5))

    for i, driver in enumerate(drivers):
        driver_stints = stint_df[stint_df["Driver"] == driver].sort_values("Stint")
        lap_cursor = 0
        for _, row in driver_stints.iterrows():
            color = COMPOUND_COLORS.get(row["Compound"], "#888888")
            ax.barh(
                i, row["NumLaps"], left=lap_cursor, color=color,
                edgecolor="black", linewidth=0.6,
            )
            lap_cursor += row["NumLaps"]

    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_xlabel("Lap")
    ax.set_title(title)
    ax.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COMPOUND_COLORS.values()]
    ax.legend(handles, COMPOUND_COLORS.keys(), loc = "upper center",
              bbox_to_anchor = (0.5, -0.35), ncol = 5, frameon=False)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi = 150, bbox_inches = "tight")
    return fig

def plot_degradation(stint_df, title = "Tire Degradation by Compound", save_path = None):
    """bar chart of mean deg rate (sec/lap) per compound."""
    summary = stint_df.groupby("Compound")["DegradationSecPerLap"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COMPOUND_COLORS.get(c, "#888888") for c in summary.index]
    bars = ax.bar(summary.index, summary.values, color=colors, edgecolor="black")
    ax.set_ylabel("Degradation (sec / lap)")
    ax.set_title(title)
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va = "bottom" if val >= 0 else "top", fontsize = 9)
    ax.axhline(0, color="black", linewidth = 0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
    
def plot_speed_trace(tel_a, tel_b, label_a, label_b, title="Speed Trace Comparison", save_path=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tel_a["Distance"], tel_a["Speed"], label = label_a, color = "#DA291C", linewidth=1.6)
    ax.plot(tel_b["Distance"], tel_b["Speed"], label = label_b, color = "#00D2BE", linewidth=1.6)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_delta_time(delta_df, label_a, label_b, title="Lap Time Delta", save_path=None):
    """
    Positive DeltaTime -> A losing time to B up to that point
    negative -> A gaining time on B
    """

    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(delta_df["Distance"], delta_df["DeltaTime"], color = "#333333", linewidth=1.6)
    ax.fill_between(delta_df["Distance"], delta_df["DeltaTime"], 0,
                    where=(delta_df["DeltaTime"] >= 0), color = "#00D2BE", alpha = 0.3,
                    label=f"{label_a} losing time")
    ax.fill_between(delta_df["Distance"], delta_df["DeltaTime"], 0,
                    where=(delta_df["DeltaTime"] <= 0), color = "#DA291C", alpha = 0.3,
                    label=f"{label_a} gaining time")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(f"Delta (s), {label_a} - {label_b}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_lap_time_evolution(laps_df, drivers, title="Race Pace Evolution", save_path=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    for driver in drivers:
        d = laps_df[laps_df["Driver"] == driver].copy()
        d["LapTimeSeconds"] = d["LapTime"].dt.total_seconds() if hasattr(d["LapTime"], "dt") else d["LapTime"]
        ax.plot(d["LapNumber"], d["LapTimeSeconds"], marker="o", markersize=3, label=driver)
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig