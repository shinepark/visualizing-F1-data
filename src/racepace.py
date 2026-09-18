"""
Race pace and tire degradation modeling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def _laps_to_seconds(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.copy()
    laps["LapTimeSeconds"]  = laps["LapTime"].dt.total_seconds()
    return laps

def compute_stint_degradation(laps: pd.DataFrame, min_laps: int = 4) -> pd.DataFrame:
    """
    Fit LapTimeSeconds ~ TyreLife per (driver, stint)

    params
    --------
    laps : DF from session.laps, must include driver, stint, compound, tyrelife, laptime
    min_laps : skip stints with fewer than this
    
    returns
    --------
    DF, one row per stint: driver, stint, compound, numlaps, degradationsecperlap, Interceptsec, R2
    """

    laps = _laps_to_seconds(laps)
    rows = []
    for (driver, stint), group in laps.groupby(["Driver", "Stint"]):
        group = group.dropna(subset=["LapTimeSeconds", "TyreLife"])
        if len(group) < min_laps:
            continue
        X = group[["TyreLife"]].values
        y = group["LapTimeSeconds"].values
        model = LinearRegression().fit(X, y)
        rows.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": group["Compound"].iloc[0],
                "NumLaps": len(group),
                "DegradationSecPerLap": round(model.coef_[0], 4),
                "InterceptSec": round(model.intercept_, 3),
                "R2": round(model.score(X, y), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["Driver", "Stint"]).reset_index(drop=True)

def compound_degradation_sum(stint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stint level deg up to compound level

    Which compound falls off fastest in this race?
    """

    return(
        stint_df.groupby("Compound")["DegradationSecPerLap"]
        .agg(["mean", "median", "std", "count"])
        .round(4)
        .sort_values("mean")
    )

def race_pace_delta(laps_a: pd.DataFrame, laps_b: pd.DataFrame) -> float:
    """
    median lap time delta (s) btwn 2 drivers' quick laps

    positive -> driver A is slower on average
    negative -> driver A is faster
    """

    a = _laps_to_seconds(laps_a)["LapTimeSeconds"].median()
    b = _laps_to_seconds(laps_b)["LapTimeSeconds"].median()
    return round(a - b, 3)
    