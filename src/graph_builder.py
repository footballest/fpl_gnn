"""
graph_builder.py
-----------------

Tools for constructing PyTorch Geometric graphs from preprocessed FPL data.

Each gameweek is modelled as an undirected graph where each node
represents a player and edges represent relationships:

* **Teammate edges** connect players on the same FPL team.  This
  allows the model to share information among teammates.
* **Opponent edges** connect players who face each other in a
  fixture.  For a match between team A and team B, every player in
  team A is connected to every player in team B to capture how
  opponent strength influences performance.

The resulting graph can be passed directly to a PyTorch Geometric
model for training or inference.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data


def build_gameweek_graph(
    gw_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "total_points",
) -> Data:
    """Convert a gameweek DataFrame into a PyTorch Geometric graph.

    Parameters
    ----------
    gw_df : pandas.DataFrame
        DataFrame containing rows for each player who played (or
        potentially available) in a given gameweek.  It must include
        columns for team identifier (`team`), opponent team
        identifier (`opponent_team`), and the chosen feature and
        label columns.  Vaastav’s merged gameweek files include
        ``team`` and ``opponent_team`` fields【781651944176581†L391-L398】.

    feature_cols : list of str
        Column names to use as node features.

    label_col : str, default "total_points"
        Column name to use as the regression target for each node.

    Returns
    -------
    torch_geometric.data.Data
        A Data object with fields:
        - ``x``: node feature matrix of shape [num_nodes, num_features]
        - ``edge_index``: tensor of shape [2, num_edges] with source and
          target node indices
        - ``y``: tensor of shape [num_nodes] containing labels (FPL points)
    """
    # Assign an index to each player row for node creation
    gw_df = gw_df.reset_index(drop=True)
    num_nodes = len(gw_df)

    # Build the node feature matrix
    features = torch.tensor(gw_df[feature_cols].values, dtype=torch.float)

    # Labels
    labels = torch.tensor(gw_df[label_col].values, dtype=torch.float)

    # Initialize lists for edge indices
    src_nodes: List[int] = []
    dst_nodes: List[int] = []

    # Precompute mapping from DataFrame index to team and opponent_team
    team_series = gw_df["team"].astype(int)
    opponent_series = gw_df["opponent_team"].astype(int)

    # Create teammate edges: connect all players with the same team
    teams = team_series.unique()
    for team_id in teams:
        indices = gw_df.index[team_series == team_id].tolist()
        # Connect each pair of teammates (i, j) with i != j
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u, v = indices[i], indices[j]
                src_nodes.extend([u, v])
                dst_nodes.extend([v, u])

    # Create opponent edges: for each fixture, connect all players on both sides
    # We identify fixtures by pairs of team and opponent_team.  For each
    # row, team is the player's club and opponent_team is the opponent.
    # We treat fixtures as undirected; thus we handle each (team, opp)
    # pair only once.
    handled_pairs = set()
    for idx, row in gw_df.iterrows():
        t = int(row["team"])
        opp = int(row["opponent_team"])
        # Use a sorted tuple to avoid duplicating (team, opponent) and (opponent, team)
        pair = tuple(sorted((t, opp)))
        if pair in handled_pairs:
            continue
        handled_pairs.add(pair)
        # Get indices of players for both teams in this fixture
        players_a = gw_df.index[team_series == t].tolist()
        players_b = gw_df.index[team_series == opp].tolist()
        for u in players_a:
            for v in players_b:
                src_nodes.extend([u, v])
                dst_nodes.extend([v, u])

    # Construct edge_index tensor
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    return Data(x=features, edge_index=edge_index, y=labels)
