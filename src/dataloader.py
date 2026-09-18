"""
FastF1 for loading session data w/ local caching.

FastF1's live-timing /Ergast-compatible backend. Requires internet access to F1's 
data services the first time a session is loaded. After that results cached locally
in `f1_cache/` and load faster.

Usage:
    from dataloader import loadsession, getdriverlaps

    session = loadsession(2023, "Bahrain", "R")
    ham_laps =getdriverlaps(session, "HAM")
"""

from pathlib import Path

import fastf1
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def loadsession(year: int, gp: str, session_type: str = "R") -> fastf1.core.Session:
    """
    Load full F1 session (laps, telem, weather, results)

    Params
    ----------
    year
    gp : Grand Prix name or round number
    session_type : "FP1", "FP2", "FP3", "Q", "S", "R"

    Returns
    ----------
    fastf1.core.Session
    """

    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

def getdriverlaps(session: fastf1.core.Session, driver_code: str) -> pd.DataFrame:
    """
    'quick' (representative, non outliers) laps for a driver
    """
    return session.laps.pick_drivers(driver_code).pick_quicklaps()

def getalllaps(session: fastf1.core.Session) -> pd.DataFrame:
    """
    every lap in session
    """
    return session.laps

def getfastestlap_telem(session: fastf1.core.Session, driver_code: str) -> pd.DataFrame:
    """
    telem (speed, throttle, brake, gear, dist) for fastest lap
    """
    lap = session.laps.pick_drivers(driver_code).pick_fastest()
    return lap.get_telemetry()

def getresults(session: fastf1.core.Session) -> pd.DataFrame:
    """
    final classification: pos, driver, team, status, points
    """
    return session.results