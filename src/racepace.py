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

def compute_stint_degradation(laps: pd.DataFrame, min_laps: int = 4,
                              fuel_effect_sec_per_lap: float = 0.0) -> pd.DataFrame:
    """
    Fit LapTimeSeconds ~ TyreLife per (driver, stint)

    params
    --------
    laps : DF from session.laps, must include driver, stint, compound, tyrelife, laptime
    min_laps : skip stints with fewer than this
    fuel_effect_sec_per_lap: within a stint, TyreLife and LapNumber increase 
    assumed fuel burn rate from outside regression
    
    returns
    --------
    DF, one row per stint: Driver, Stint, Compound, NumLaps,
    DegradationSecPerLap (fuel-corrected if requested), RawSlopeSecPerLap
    (always the uncorrected regression output, for comparison),
    InterceptSec, R2
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
        raw_slope = model.coef_[0]
        rows.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": group["Compound"].iloc[0],
                "NumLaps": len(group),
                "DegradationSecPerLap": round(raw_slope + fuel_effect_sec_per_lap, 4),
                "RawSlopeSecPerLap": round(raw_slope, 4),
                "InterceptSec": round(model.intercept_, 3),
                "R2": round(model.score(X, y), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["Driver", "Stint"]).reset_index(drop=True)

def compound_model_from_stints(stint_df: pd.DataFrame, pace_offsets: dict | None = None,
                               verbose: bool = True) -> dict:
    """
    convert fitted stint level deg into compound_model format expected so Monte Carlo strat
    simulation is driven by degradation rates actually fitted from session data
    """
    pace_offsets = pace_offsets or {}
    summary = stint_df.groupby("Compound")["DegradationSecPerLap"].agg(["mean", "std", "count"])
    model = {}
    for compound, row in summary.iterrows():
        raw_mean = row["mean"]
        deg_mean = max(0.0, raw_mean)
        if verbose and raw_mean < 0:
            print(f"  [note] {compound}: fitted mean degradation was {raw_mean:+.4f} s/lap "
                  f"(negative likely fuel effect outweighing tire wear at this circuit); "
                  f"flooring to 0.0 for the simulator. Pass fuel_effect_sec_per_lap to "
                  f"compute_stint_degradation() to correct for this instead of flooring.")
        # only one stint -> no std to estimate
        deg_std = row["std"] if (row["count"] > 1 and not np.isnan(row["std"])) else max(deg_mean * 0.2, 0.005)
        model[compound] = {
            "pace_offset": pace_offsets.get(compound, 0.0),
            "deg_mean": deg_mean,
            "deg_std": deg_std,
        }
    return model

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
    