"""
train.py
--------

Command-line interface for training the FPL Graph Neural Network.

This script loads historical FPL data (either from the Vaastav CSVs
or via the official API), constructs graphs for each gameweek
between ``start_gw`` and ``end_gw``, trains a Graph Attention Network
to predict player points and prints basic training/validation metrics.

Example usage::

    python src/train.py --season 2023-24 --start_gw 1 --end_gw 30

Requirements: See ``requirements.txt`` for package dependencies.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import Adam

from data_loader import (
    load_vaastav_merged_gw,
    load_vaastav_gameweek,
    add_rolling_form,
)
from graph_builder import build_gameweek_graph
from model import FPLGNN


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FPL GNN")
    parser.add_argument("--season", type=str, required=True, help="Season name, e.g. 2023-24")
    parser.add_argument("--start_gw", type=int, required=True, help="First gameweek to include")
    parser.add_argument("--end_gw", type=int, required=True, help="Last gameweek to include")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for GAT")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads in first GAT layer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args(args)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    season = args.season
    start_gw = args.start_gw
    end_gw = args.end_gw

    # Load merged season data to compute rolling form
    season_df = load_vaastav_merged_gw(season)
    season_df = add_rolling_form(season_df, window=3)

    # Determine candidate feature columns.  We select a handful of
    # numerical columns commonly available in the FPL data.  Use the
    # intersection to avoid missing columns.
    candidate_features = [
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "saves" if "saves" in season_df.columns else None,
        "yellow_cards" if "yellow_cards" in season_df.columns else None,
        "red_cards" if "red_cards" in season_df.columns else None,
        "form",
    ]
    # Filter out None values and columns not in DataFrame
    feature_cols = [c for c in candidate_features if c and c in season_df.columns]

    print(f"Using feature columns: {feature_cols}")

    # Build graph objects for each gameweek
    graphs = []
    for gw in range(start_gw, end_gw + 1):
        gw_df = load_vaastav_gameweek(season, gw)
        # Merge form from season_df into gw_df by matching (element, round)
        # Some columns may not be present in gw_df, so we merge on id and round
        merge_cols = ["element", "round"]
        gw_df = gw_df.merge(
            season_df[merge_cols + ["form"]], on=merge_cols, how="left", suffixes=("", "_season"),
        )
        # Fill missing form with 0
        gw_df["form"] = gw_df["form"].fillna(0)

        # Build graph
        graph = build_gameweek_graph(gw_df, feature_cols=feature_cols, label_col="total_points")
        graphs.append(graph)
        print(f"Built graph for GW{gw} with {graph.x.size(0)} nodes and {graph.edge_index.size(1)} edges")

    # Split into train and validation sets (e.g., last 20% for validation)
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]

    # Create model
    in_features = len(feature_cols)
    model = FPLGNN(in_features=in_features, hidden_features=args.hidden_dim, heads=args.heads)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for graph in train_graphs:
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index).squeeze()
            loss = F.mse_loss(out, graph.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_graphs)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph in val_graphs:
                out = model(graph.x, graph.edge_index).squeeze()
                loss = F.mse_loss(out, graph.y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_graphs) if val_graphs else float('nan')
        print(f"Epoch {epoch:3d}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
