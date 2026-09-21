"""
compares pit stop strats for race using Monte Carlo simulation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import racepace
import strategy_optimizer as so
import visualize

OUT = Path(__file__).parent / "outputs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

# fresh tire pace adv isn't identifiable from degrad reg alone
# supplied from known real world compound pace deltas
PACE_OFFSETS = {"SOFT": -0.55, "MEDIUM": -0.25, "HARD": 0.0}

def get_stint_df_and_race_laps():
    """
    returns (stint_df, race_laps)
    """
    if len(sys.argv) == 4:
        import dataloader

        year = int(sys.argv[1])
        gp = sys.argv[2]
        session_type = sys.argv[3]

        print(f"Loading {year} {gp} {session_type} ...")
        session = dataloader.loadsession(year, gp, session_type)

        #use every driver's laps
        #more stints per compound makes deg fit more stable
        laps = dataloader.getalllaps(session)
        stint_df = racepace.compute_stint_degradation(laps)
        race_laps = int(laps["LapNumber"].max())
        return stint_df, race_laps
    
    else:
        print(__doc__)
        sys.exit(1)

def main():
    stint_df, race_laps = get_stint_df_and_race_laps()
    print("=== Fitted stint degradation (input to the compoud model) ===")
    print(stint_df.to_string(index=False))

    fitted_compound_model = racepace.compound_model_from_stints(stint_df, PACE_OFFSETS)

    print("\n=== Compound model derived from fitted stint degradation ===")
    for compound, params in fitted_compound_model.items():
        print(f" {compound:8s} deg_mean={params['deg_mean']:.4f} "
              f"deg_std={params['deg_std']:.4f} pace_offset={params['pace_offset']:+.2f}")
    
    missing = {"SOFT", "MEDIUM", "HARD"} - set(fitted_compound_model.keys())
    if missing:
        print(f"\nNo laps found on {missing} in this session"
              f"strategies using {missing} will fail. Pick strategies that "
              f"only use compounds actually run in this race or add fallback values.")
    print()

    cfg = so.RaceConfig(race_laps=race_laps, compound_model=fitted_compound_model)

    n = race_laps
    strategies = {
        f"1-stop (M{n//3}/H{n - n//3})":    [("MEDIUM", n // 3), ("HARD", n - n // 3)],
        f"1-stop early (M{n//5}/H{n - n//5})":    [("MEDIUM", n // 5), ("HARD", n - n // 5)],
        f"2-stop (S{n//4}/M{n//4}/H{n - 2*(n//4)})": [
            ("SOFT", n // 4), ("MEDIUM", n // 4), ("HARD", n - 2 * (n // 4))
        ],
    }

    print(f"simulating {len(strategies)} strategies over {race_laps} laps "
          f"(5000 Monte Carlo iterations each, shared race conditions)... \n")

    sims_df, summary_df = so.simulate_strategies(strategies, cfg, n_sims = 5000, seed = 42)

    print(summary_df.to_string(index=False))
    print(
        "\nMedianTime/P10/P90 in seconds. WinProbability = fraction of the "
        "5000 simulated races in which this strategy produced the lowest "
        "total time, holding lap noise and safety-car timing identical "
        "across strategies within each simulated race."
    )

    fastest = summary_df.iloc[0]
    slowest = summary_df.iloc[-1]
    gap = slowest["MedianTime"] - fastest["MedianTime"]
    print(
        f"\n'{fastest.Strategy}' beats '{slowest.Strategy}' by a median of "
        f"{gap:.1f}s over the race and wins outright in "
        f"{fastest.WinProbability:.0%} of simulated realizations."
    )

    visualize.plot_strategy_distributions(
        sims_df, summary_df, save_path=OUT / "06_strat_distributions.png"
    )

    visualize.plot_win_probability(
        summary_df, save_path=OUT / "07_strat_win_prob.png"
    )

if __name__ == "__main__":
    main()


