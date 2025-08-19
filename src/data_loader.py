"""
data_loader.py
----------------

Utility functions for acquiring and preprocessing FPL data.

This module provides helpers to download FPL gameweek data from the
open-source **Fantasy‑Premier‑League** dataset maintained by Vaastav and
to fetch live data from the official Fantasy Premier League API.  The
Vaastav dataset is organised by season with CSV files for each
gameweek and a merged file of all gameweeks【781651944176581†L391-L398】.  The official API exposes endpoints
such as `bootstrap-static`, `fixtures`, `element-summary` and `live`
that return JSON data【628285275049102†L263-L344】.

The functions in this file do minimal preprocessing; the heavy
feature engineering logic should live in `utils.py`.
"""

from __future__ import annotations

import pandas as pd
import requests
from typing import Dict, Any, Optional


def load_vaastav_merged_gw(season: str) -> pd.DataFrame:
    """Load the merged gameweek CSV for a given season.

    Parameters
    ----------
    season : str
        Season name, e.g. ``"2023-24"``.  This should correspond to a
        folder name in the Vaastav dataset.  See the repository’s data
        directory structure【781651944176581†L391-L398】.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing gameweek statistics for all players
        throughout the season.

    Notes
    -----
    The CSV is loaded from the raw GitHub URL.  If you need to
    download and cache it locally, call :func:`download_file` in
    `utils.py`.
    """
    url = (
        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season}/gws/merged_gw.csv"
    )
    df = pd.read_csv(url)
    return df


def load_vaastav_gameweek(season: str, gw: int) -> pd.DataFrame:
    """Load a single gameweek CSV for a given season and gameweek number.

    Parameters
    ----------
    season : str
        Season name (e.g. ``"2023-24"``).
    gw : int
        Gameweek number starting from 1.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing statistics for the specified gameweek.
    """
    url = (
        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season}/gws/gw{gw}.csv"
    )
    return pd.read_csv(url)


def fetch_fpl_api_bootstrap(session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch the bootstrap-static endpoint from the official FPL API.

    The bootstrap endpoint returns metadata such as player information,
    team IDs and current gameweek status.  This wrapper returns the
    parsed JSON as a Python dict【628285275049102†L263-L290】.

    Parameters
    ----------
    session : requests.Session, optional
        Optional session object to reuse connections.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    s = session or requests.Session()
    resp = s.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_api_element_summary(player_id: int, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch a player's history using the element-summary endpoint.

    Parameters
    ----------
    player_id : int
        The unique FPL element identifier for the player.
    session : requests.Session, optional
        Optional session for connection reuse.

    Returns
    -------
    dict
        JSON response containing fields ``fixtures``, ``history`` and
        ``history_past``【628285275049102†L292-L299】.
    """
    s = session or requests.Session()
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    resp = s.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_api_fixtures(event_id: Optional[int] = None, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch fixtures from the FPL API.

    If ``event_id`` is provided, only fixtures for that gameweek are returned;
    otherwise all fixtures are returned【628285275049102†L323-L333】.

    Parameters
    ----------
    event_id : int, optional
        Gameweek identifier.
    session : requests.Session, optional
        Optional HTTP session.

    Returns
    -------
    dict
        JSON object representing the fixtures.
    """
    s = session or requests.Session()
    url = "https://fantasy.premierleague.com/api/fixtures/"
    if event_id is not None:
        url = f"{url}?event={event_id}"
    resp = s.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl_api_live(event_id: int, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch live data for a given gameweek.

    The live endpoint returns per-player scores and bonus points as the
    gameweek progresses【628285275049102†L334-L344】.

    Parameters
    ----------
    event_id : int
        Gameweek identifier.
    session : requests.Session, optional
        Optional HTTP session.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    s = session or requests.Session()
    url = f"https://fantasy.premierleague.com/api/event/{event_id}/live/"
    resp = s.get(url)
    resp.raise_for_status()
    return resp.json()


def add_rolling_form(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute a rolling average of total points to represent form.

    This helper takes a DataFrame containing at least the columns
    ``id`` (player identifier), ``round`` (gameweek number) and
    ``total_points`` (points scored in the gameweek).  It returns the
    input DataFrame with an additional column ``form`` that is the
    mean of the player's points over the previous ``window`` gameweeks.
    Missing values are filled with zeros.

    Parameters
    ----------
    df : pandas.DataFrame
        Gameweek data for a season, such as the output of
        :func:`load_vaastav_merged_gw`.
    window : int, default 3
        Number of past gameweeks to include in the rolling average.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new ``form`` column.
    """
    # Sort by player and gameweek to ensure correct rolling computation
    df_sorted = df.sort_values(["element", "round"]).copy()
    df_sorted["form"] = (
        df_sorted.groupby("element")["total_points"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df_sorted
