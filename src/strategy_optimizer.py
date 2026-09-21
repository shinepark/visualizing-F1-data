"""
monte carlo comparison of pit stop strats
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# compound model: pace_offset is seconds faster than baseline on fresh tire
# deg_mean/deg_std are seconds per lap degradation
DEFAULT_COMPOUND_MODEL = {
    "SOFT": {"pace_offset": -0.55, "deg_mean": 0.090, "deg_std": 0.015},
    "MEDIUM": {"pace_offset": -0.25, "deg_mean": 0.050, "deg_std": 0.010},
    "HARD": {"pace_offset": 0.00, "deg_mean": 0.020, "deg_std": 0.006},
}

@dataclass
class RaceConfig:
    race_laps: int
    base_pace: float = 92.0 # seconds, "neutral" compound free lap
    lap_noise_std: float = 0.12 # per lap random variation
    pit_loss_mean: float = 22.0 # seconds lost per green flag stop
    pit_loss_std: float = 1.5
    sc_lambda: float = 0.9 # expected # of safety car periods per race
    sc_window: int = 5 # laps a SC period lasts
    sc_pit_loss_multiplier: float = 0.35 # pitting under SC costs this fraction of normal
    compound_model: dict = field(default_factory=lambda: DEFAULT_COMPOUND_MODEL)

def _validate_strategy(strategy: list[tuple[str, int]], race_laps: int, name: str):
    total = sum(n for _, n in strategy)
    if total != race_laps:
        raise ValueError(
            f"Strategy '{name}' covers {total} laps, expected {race_laps}."
        )
    compounds_used = {c for c, _ in strategy}
    if len(strategy) > 1 and len(compounds_used) < 2:
        # have to use 2 different compounds
        raise ValueError(
            f"Strategy '{name}' uses only one compound across multiple "
            f"stints ({compounds_used}) -- not legal in a dry race."
        )
    
def _sample_safety_car_windows(cfg: RaceConfig, rng: np.random.Generator):
    """returns bool array of length race_laps: true where a SC is active."""
    n_periods = rng.poisson(cfg.sc_lambda)
    sc_active = np.zeros(cfg.race_laps + 1, dtype=bool) #1-indexed convenience
    for _ in range(n_periods):
        start = rng.integers(1, max(cfg.race_laps - cfg.sc_window, 2))
        end = min(start + cfg.sc_window, cfg.race_laps)
        sc_active[start:end + 1] = True
    return sc_active

def _simulate_one_strategy(
    strategy: list[tuple[str, int]],
    cfg: RaceConfig,
    lap_noise: np.ndarray,
    sc_active: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Total race time (s) for 1 strat oner one shared race realization."""
    total_time = 0.0
    lap = 0
    n_stints = len(strategy)

    for stint_idx, (compound, n_laps) in enumerate(strategy):
        model = cfg.compound_model[compound]
        # same this stint's deg slope once (race to race variation in how compound behaves) not per lap
        deg_slope = max(0.0, rng.normal(model["deg_mean"], model["deg_std"]))

        for tyre_life in range(1, n_laps + 1):
            lap += 1
            lap_time = (
                cfg.base_pace
                + model["pace_offset"]
                + deg_slope * tyre_life
                + lap_noise[lap - 1]
            )
            total_time += lap_time

        # pit stop after this sting, unless it's the final stint
        if stint_idx < n_stints - 1:
            pit_loss = max(3.0, rng.normal(cfg.pit_loss_mean, cfg.pit_loss_std))
            if sc_active[lap]:
                pit_loss *= cfg.sc_pit_loss_multiplier
            total_time += pit_loss

    return total_time

def simulate_strategies(
    strategies: dict[str, list[tuple[str, int]]],
    cfg: RaceConfig,
    n_sims: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    run n_sims paired Monte Carlo races and compare every strat under same shared conditions each iter

    returns
    --------
    sims_df: one row per (simulation, strat) w/TotalTime
    summary_df: one row per strat w/ median/p10/p90 and the fraction
    of simulations in which it was fastest ('win prob')
    """

    for name, strat in strategies.items():
        _validate_strategy(strat, cfg.race_laps, name)

    rng = np.random.default_rng(seed)
    names = list(strategies.keys())
    records = []
    wins = {name: 0 for name in names}

    for sim in range(n_sims):
        #shared race realization across all strats this iteration
        lap_noise = rng.normal(0, cfg.lap_noise_std, cfg.race_laps)
        sc_active = _sample_safety_car_windows(cfg, rng)

        times = {}
        for name in names:
            t = _simulate_one_strategy(strategies[name], cfg, lap_noise, sc_active, rng)
            times[name] = t
            records.append({"Simulation": sim, "Strategy": name, "TotalTime": t})

        winner = min(times, key=times.get)
        wins[winner] += 1

    sims_df = pd.DataFrame(records)

    summary_rows = []
    for name in names:
        times = sims_df.loc[sims_df.Strategy == name, "TotalTime"]
        summary_rows.append(
            {
                "Strategy": name,
                "MedianTime": round(times.median(), 2),
                "P10": round(times.quantile(0.10), 2),
                "P90": round(times.quantile(0.90), 2),
                "StdDev": round(times.std(), 2),
                "WinProbability": round(wins[name] / n_sims, 3),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("MedianTime").reset_index(drop=True)

    return sims_df, summary_df
