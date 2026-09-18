"""
runs analysis pipeline on a grand prix session pulled via FastF1

Usage:
    python runsession.py 2023 Bahrain R VER HAM
    python runsession.py 2024 Monza R LEC SAI
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import dataloader
import racepace
import telemanalysis
import visualize

OUT = Path(__file__).parent / "outputs" / "charts_real"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)

    year = int(sys.argv[1])
    gp = sys.argv[2]
    session_type = sys.argv[3]
    driver_a = sys.argv[4].upper()
    driver_b = sys.argv[5].upper()

    print(f"Loading {year} {gp} {session_type} ... (first run will hit the network)")
    session = dataloader.loadsession(year, gp, session_type)

    laps = dataloader.getalllaps(session)
    laps = laps[laps["Driver"].isin([driver_a, driver_b])]

    # tire deg
    stint_df = racepace.compute_stint_degradation(laps)
    print("\n=== Stint level degradation ===")
    print(stint_df.to_string(index=False))
    print("\n=== Degradation by compound ===")
    print(racepace.compound_degradation_sum(stint_df).to_string())

    delta = racepace.race_pace_delta(
        dataloader.getdriverlaps(session, driver_a),
        dataloader.getdriverlaps(session, driver_b),
    )

    print(f"\nMedian race pace delta ({driver_a} - {driver_b}): {delta:+.3f} s/lap")
    
    visualize.plot_stint_strategy(stint_df, save_path=OUT / "01_tire_strat.png")
    visualize.plot_degradation(stint_df, save_path=OUT / "02_deg_by_compound.png")
    visualize.plot_lap_time_evolution(
        laps, drivers=[driver_a, driver_b],
        title=f"{year} {gp} {session_type} - Race Pace Evolution",
        save_path=OUT / "03_lap_time_evolution.png",
    )

    # telem : fastest lap comparison
    tel_a = dataloader.getfastestlap_telem(session, driver_a)
    tel_b = dataloader.getfastestlap_telem(session, driver_b)

    visualize.plot_speed_trace(
        tel_a, tel_b, driver_a, driver_b,
        title=f"{year} {gp} {session_type} - Fastest Lap Speed Trace",
        save_path=OUT / "04_speed_trace.png",
    )

    delta_df = telemanalysis.compute_delta_time(tel_a, tel_b)
    visualize.plot_delta_time(
        delta_df, driver_a, driver_b,
        title=f"Lap Time Delta: {driver_a} vs {driver_b}",
        save_path=OUT / "05_delta_time.png",
    )

if __name__ == "__main__":
    main()