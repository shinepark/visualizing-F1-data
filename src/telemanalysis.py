"""
corner by corner comparison btwn 2 drivers using channel telem
synced to distance around lap plus a running time delta trace
that shows exactly where one driver gains/loses time
"""

import numpy as np
import pandas as pd

def interpolate_to_common_distance(tel: pd.DataFrame, n_points: int = 1000) -> pd.DataFrame:
    """
    resample telem trace onto a fixed distace grid (0 -> max dist) so
    2 diff drivers' laps can be compared point for point.
    """

    dist = np.linspace(tel["Distance"].min(), tel["Distance"].max(), n_points)
    out = pd.DataFrame({"Distance": dist})
    for col in ["Speed", "Throttle", "Brake", "nGear", "RPM"]:
        if col in tel.columns:
            out[col] = np.interp(dist, tel["Distance"], tel[col])
    return out

def compute_delta_time(tel_a: pd.DataFrame, tel_b: pd.DataFrame, n_points: int = 1000) -> pd.DataFrame:
    """
    approx. cumulative time delta btwn driver A/B across a lap

    returns DF w/ distance and delta time (seconds, A minus B;
    negative means A is ahead of B at that point on track)
    """

    a = interpolate_to_common_distance(tel_a, n_points)
    b = interpolate_to_common_distance(tel_b, n_points)

    # avoid dividing by zero on pit lane / standing starts
    speed_a = np.clip(a["Speed"].values, 1, None) / 3.6 # km/h -> m/s
    speed_b = np.clip(b["Speed"].values, 1, None) / 3.6

    dist_step = np.diff(a["Distance"].values, prepend=a["Distance"].values[0])
    time_a = np.cumsum(dist_step / speed_a)
    time_b = np.cumsum(dist_step / speed_b)

    return pd.DataFrame(
        {
            "Distance": a["Distance"],
            "DeltaTime": time_a - time_b,
            "Speed_A": a["Speed"],
            "Speed_B": b["Speed"],
        }
    )

def indentify_gain_loss_zones(delta_df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    find contiguous track sections where delta time slope indicates one driver is
    consistently gaining/losing time
    """

    d = delta_df.copy()
    d["Slope"] = np.gradient(d["DeltaTime"], d["Distance"])
    d["Zone"] = np.where(
        d["Slope"] > threshold, "A_losing",
        np.where(d["Slope"] < -threshold, "A_gaining", "even"),
    )
    return d