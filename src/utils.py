"""
utils.py
--------

Helper functions for the FPL GNN project.

This module contains miscellaneous utilities used across the project,
such as downloading files, computing derived features and evaluating
model performance.  Adding more utilities here keeps other modules
focused on a single responsibility.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

import pandas as pd


def download_file(url: str, dest_path: str, overwrite: bool = False) -> str:
    """Download a file from a URL to a local path.

    If ``overwrite`` is False and the destination file already exists,
    the download is skipped.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    dest_path : str
        Local filesystem path where the file should be saved.
    overwrite : bool, default False
        Whether to overwrite an existing file.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    if os.path.exists(dest_path) and not overwrite:
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def compute_team_strengths(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple team strength metrics from fixtures.

    This placeholder demonstrates how to derive additional team-level
    features.  It expects a DataFrame with at least columns
    ``team_a_score`` and ``team_h_score`` representing match scores.  It
    returns a summary table with goals scored and conceded per team.

    Parameters
    ----------
    fixtures_df : pandas.DataFrame
        Fixture data, e.g., from the FPL API or an external source.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by team id with columns ``goals_for`` and
        ``goals_against``.
    """
    # Example implementation: sum goals for and against across fixtures
    stats = {
        'goals_for': {},
        'goals_against': {},
    }
    for _, row in fixtures_df.iterrows():
        home_team = row.get('team_h')
        away_team = row.get('team_a')
        home_goals = row.get('team_h_score', 0)
        away_goals = row.get('team_a_score', 0)
        if home_team is not None:
            stats['goals_for'][home_team] = stats['goals_for'].get(home_team, 0) + home_goals
            stats['goals_against'][home_team] = stats['goals_against'].get(home_team, 0) + away_goals
        if away_team is not None:
            stats['goals_for'][away_team] = stats['goals_for'].get(away_team, 0) + away_goals
            stats['goals_against'][away_team] = stats['goals_against'].get(away_team, 0) + home_goals
    result = pd.DataFrame(stats)
    result.index.name = 'team_id'
    return result
