"""
compares pit stop strats for race using Monte Carlo simulation

Usage:
    python run_optimizer.py 2023 Bahrain R
    python run_optimizer.py 2024 Monza R
    python run_optimizer.py 2024 Monza R --fuel-correct 0.05
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import dataloader
import racepace
import strategy_optimizer as so
import visualize

OUT = Path(__file__).parent / "outputs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# fresh tire pace adv isn't identifiable from degrad reg alone
# supplied from known real world compound pace deltas
PACE_OFFSETS = {"SOFT": -0.55, "MEDIUM": -0.25, "HARD": 0.0}

#c ompounds ordered fastest fresh to slowest fresh
# used to build sensible strat candidates out of whatever compounds avail
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]

def _parse_args():
    args = sys.argv[1:]
    fuel_correction = 0.0
    if "--fuel-correct" in args:
        idx = args.index("--fuel-correct")
        fuel_correction = float(args[idx + 1])
        del args[idx:idx + 2]
    if len(args) != 3:
        print(__doc__)
        sys.exit(1)
    return args, fuel_correction

def load_real_stint_df(year: int, gp: str, session_type: str, fuel_correction: float):
    """
    load real session and fit stint level tire deg from laps
    """
    print(f"Loading {year} {gp} {session_type} ...")
    session = dataloader.loadsession(year, gp, session_type)

    #use every driver's laps
    #more stints per compound makes deg fit more stable

    laps = dataloader.getalllaps(session)
    stint_df = racepace.compute_stint_degradation(laps, fuel_effect_sec_per_lap=fuel_correction)
    race_laps = int(laps["LapNumber"].max())
    return stint_df, race_laps

def build_candidate_strategies(available_compounds: set, race_laps: int) -> dict:
    """
    build a set of 1 stop and 2 stop strats using only compounds actually in fitted model

    requires at least 2 available compounds (a dry race needs to use >=2)
    """

    avail = [c for c in COMPOUND_ORDER if c in available_compounds]
    if len(avail) < 2:
        raise ValueError(
            f"Only {avail} has fitted degradation data, need at least 2 compounds"
            f" to have a legal multi stint strat."
        )

    fastest, hardest = avail[0], avail[-1]
    n = race_laps
    strategies = {}

    for label, first_stint_laps in [("early", n // 5), ("mid", n // 3), ("late", 2 * n // 3)]:
        s1 = max(1, min(first_stint_laps, n - 1))
        s2 = n - s1
        strategies[f"1-stop {label} ({fastest[0]}{s1}/{hardest[0]}{s2})"] = [
            (fastest, s1), (hardest, s2)
        ]

    third = max(1, n // 3)
    remainder = n - 2 * third
    if remainder < 1:
        third = max(1, n // 3 - 1)
        remainder = n - 2 * third

    if len(avail) >= 3:
        middle = avail[1]
        strategies[f"2-stop ({fastest[0]}{third}/{middle[0]}{third}/{hardest[0]}{remainder})"] = [
            (fastest, third), (middle, third), (hardest, remainder)
        ]
    else:
        # only 2 compounds avail
        strategies[f"2-stop ({fastest[0]}{third}/{fastest[0]}{third}/{hardest[0]}{remainder})"] = [
            (fastest, third), (fastest, third), (hardest, remainder)
        ]

    return strategies

def main():
    args, fuel_correction = _parse_args()
    year, gp, session_type = int(args[0]), args[1], args[2]

    stint_df, race_laps = load_real_stint_df(year, gp, session_type, fuel_correction)

    print(f"\n=== Fitted stint degradation: {year} {gp} {session_type} ===")
    cols = ["Driver", "Stint", "Compound", "NumLaps", "DegradationSecPerLap", "RawSlopeSecPerLap", "R2"]
    print(stint_df[cols].to_string(index=False))
    if fuel_correction:
        print(f"\n(DegradationSecPerLap includes a +{fuel_correction:.3f} s/lap fuel-burn "
              f"correction; RawSlopeSecPerLap is the uncorrected regression output.)")

    print("\n=== Compound model derived from fitted stint degradation ===")
    fitted_compound_model = racepace.compound_model_from_stints(stint_df, PACE_OFFSETS)
    for compound, params in fitted_compound_model.items():
        print(f"  {compound:8s}  deg_mean={params['deg_mean']:.4f}  "
              f"deg_std={params['deg_std']:.4f}  pace_offset={params['pace_offset']:+.2f}")
    print()

    strategies = build_candidate_strategies(set(fitted_compound_model.keys()), race_laps)
    print(f"Built {len(strategies)} candidate strategies from available compounds "
          f"{sorted(fitted_compound_model.keys())}:")
    for name, stints in strategies.items():
        print(f"  {name}: {stints}")

    cfg = so.RaceConfig(race_laps=race_laps, compound_model=fitted_compound_model)

    print(f"\nSimulating {len(strategies)} strats over {race_laps} laps "
          f"(5000 Monte Carlo iterations each) \n")

    sims_df, summary_df = so.simulate_strategies(strategies, cfg, n_sims = 5000, seed = 42)


    print(summary_df.to_string(index=False))
    print(
        "\nMedianTime/P10/P90 in seconds. WinProbability = fraction of the "
        "5000 simulated races in which this strategy produced the lowest "
        "total time, holding lap noise and safety-car timing identical "
        "across strategies within each simulated race."
    )

    fastest_strat = summary_df.iloc[0]
    slowest_strat = summary_df.iloc[-1]
    gap = slowest_strat["MedianTime"] - fastest_strat["MedianTime"]
    print(
        f"\n'{fastest_strat.Strategy}' beats '{slowest_strat.Strategy}' by a median of "
        f"{gap:.1f}s over the race -- and wins outright in "
        f"{fastest_strat.WinProbability:.0%} of simulated realizations."
    )

    visualize.plot_strategy_distributions(
        sims_df, summary_df,
        title=f"{year} {gp} {session_type} — Strategy Comparison",
        save_path=OUT / "06_strategy_distributions.png",
    )
    visualize.plot_win_probability(
        summary_df,
        title=f"{year} {gp} {session_type} — Win Probability by Strategy",
        save_path=OUT / "07_strategy_win_probability.png",
    )


if __name__ == "__main__":
    main()


