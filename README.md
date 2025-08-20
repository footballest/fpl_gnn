![CI](https://github.com/footballest/fpl_gnn/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# Graph Neural Network for FPL Score Prediction

This repository contains code to build a **graph neural network** (GNN) that predicts Fantasy Premier League (FPL) player scores for upcoming gameweeks.  Instead of treating players in isolation, the model represents each gameweek as a graph where **players are nodes and edges represent relationships** (teammate and opponent connections).  This design is inspired by football analytics research which argues that a player's output is influenced by both their own form and the interactions with teammates and opponents.  The project uses a **Graph Attention Network (GAT)** implemented with PyTorch Geometric to learn which connections are most influential on a player’s expected points.

## Data sources

We rely on publicly available FPL data.  There are two common ways to obtain historical gameweek statistics:

* **Vaastav’s Fantasy‑Premier‑League dataset** – An open‑source GitHub repository that compiles CSV files for each season.  The data directory is organized by season; for example, `season/gws/gw_number.csv` contains game‑week‑specific stats and `season/gws/merged_gws.csv` merges all gameweeks for that season.  Because weekly updates stopped after the 2024–25 season, the repository now publishes three updates per season (start of season, post‑January window and end of season).  Nevertheless, it provides complete historical data that can be read directly in Python via `pandas.read_csv` using raw GitHub URLs.

* **Official FPL API** – The official game exposes several JSON endpoints.  Unofficial libraries such as [`fpl-api`](https://github.com/jeppe-smith/fpl-api) wrap these endpoints and provide typed functions:
  * `fetchBootstrap()` returns general information on players, teams and gameweeks.
  * `fetchElementSummary(playerId)` returns a specific player’s history.
  * `fetchFixtures(eventId)` returns fixtures for all gameweeks or a given gameweek】.
  * `fetchLive(eventId)` returns live scores for a gameweek.

For training you can either download the CSV files into the `data/` folder or query the API and save the responses as needed.

## Project structure

```text
fpl-gnn-project/
├── README.md              # Project overview and setup instructions
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore large or temporary files
├── data/                  # (Optional) place to store downloaded CSV data
├── notebooks/             # Jupyter notebooks for exploration and data prep
│   └── EDA_and_DataPrep.ipynb
├── src/
│   ├── data_loader.py     # Functions to download and preprocess FPL data
│   ├── graph_builder.py   # Convert preprocessed data into PyTorch Geometric graphs
│   ├── model.py           # Definition of the GNN model using GATConv
│   ├── train.py           # Training loop and evaluation code
│   └── utils.py           # Helper functions (e.g., computing rolling averages)
└── docs/                  # Placeholder for documentation and visualizations
```

## Setup

1. **Clone the repository** and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The project uses `pandas`, `numpy`, `torch` and `torch_geometric`.  Installing PyTorch Geometric requires matching the correct version of PyTorch and CUDA; refer to the official installation guide if needed.

2. **Prepare data** by either downloading the Vaastav CSV files into the `data/` directory or writing a script that calls the FPL API endpoints.  The `data_loader.py` module provides example functions to load CSV data via pandas and to fetch JSON from the API.

3. **Train the model** using:

   ```bash
   python src/train.py --season 2023-24 --start_gw 1 --end_gw 30
   ```

   The script loads the specified season, constructs a graph for each gameweek, trains the GAT model to predict player points and prints training/validation metrics.  Adjust the season and gameweek range as needed.

## Model overview

The model defined in `src/model.py` uses two **Graph Attention Network** layers.  The first GAT layer has multiple attention heads to learn different aspects of player interactions, and the second layer outputs a single regression value per node (player).  PyTorch Geometric provides the `GATConv` layer, which implements the attention mechanism described in the original paper.  During training we minimize mean squared error between predicted and actual FPL points.

## Future work

Potential enhancements include:

* **Temporal modeling** – Use recurrent or convolutional neural networks to better capture how player form evolves over successive gameweeks.
* **Heterogeneous edges** – Differentiate between teammate and opponent connections or connect only specific positions to reflect on‑field interactions.
* **Interpretability** – Extract attention weights and visualize which teammates or opponents influence a prediction.  These could be shown as graph diagrams or heatmaps in the `docs/` folder.

---

This project is under active development.  Check the source code and notebooks for examples and updates.
