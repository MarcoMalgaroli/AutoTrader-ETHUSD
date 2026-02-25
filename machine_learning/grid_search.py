"""
Grid Search – Walk-Forward Hyperparameter Tuning for LSTM Classifier and MLP.

Each hyperparameter combination is evaluated by running the **full walk-forward
backtest** (the same one used in ``backtest_triple_barrier``).  The scoring
metric is ``final_equity``, which is what actually matters.

This is slower than a single-split validation search, so the default grids are
intentionally small.  Expand them once you have a rough sense of the best region.

Usage (from project root):
    python -m machine_learning.grid_search
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import machine_learning.lstm_classifier as lstm_mod
import machine_learning.mlp as mlp_mod
from backtest.backtest_triple_barrier import backtest_triple_barrier

with open(Path(__file__).resolve().parent.parent / "config.json", "r") as f:
    CONFIG = json.load(f)

PRINT_WIDTH = CONFIG["print_width"]


# ═══════════════════════════════════════════════════════════════════
#  HELPERS – temporarily override module-level globals
# ═══════════════════════════════════════════════════════════════════

def _patch_lstm_globals(params: dict):
    """Monkey-patch lstm_classifier module globals for one grid combo."""
    mapping = {
        "hidden_size":   "HIDDEN_SIZE",
        "num_layers":    "NUM_LAYERS",
        "learning_rate": "LEARNING_RATE",
        "dropout":       "DROPOUT",
        "seq_len":       "SEQ_LEN",
        "batch_size":    "BATCH_SIZE",
    }
    saved = {}
    for key, attr in mapping.items():
        saved[attr] = getattr(lstm_mod, attr)
        if key in params:
            setattr(lstm_mod, attr, params[key])
    return saved


def _restore_lstm_globals(saved: dict):
    for attr, val in saved.items():
        setattr(lstm_mod, attr, val)


def _patch_mlp_globals(params: dict):
    """Monkey-patch mlp module globals for one grid combo."""
    mapping = {
        "hidden_sizes":  "HIDDEN_SIZES",
        "learning_rate": "LEARNING_RATE",
        "dropout":       "DROPOUT",
        "batch_size":    "BATCH_SIZE",
    }
    saved = {}
    for key, attr in mapping.items():
        saved[attr] = getattr(mlp_mod, attr)
        if key in params:
            setattr(mlp_mod, attr, params[key])
    return saved


def _restore_mlp_globals(saved: dict):
    for attr, val in saved.items():
        setattr(mlp_mod, attr, val)


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def grid_search_lstm(
    df: pd.DataFrame,
    param_grid: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward grid search for the LSTM classifier.

    Each combo runs the full ``backtest_triple_barrier`` with model_type="lstm".
    Results are sorted by **final_equity** (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Full enhanced dataset (with features & target already computed).
    param_grid : dict, optional
        Keys: hidden_size, num_layers, learning_rate, dropout, seq_len, batch_size.
        Values are lists of candidates.  A sensible small default is provided.
    seed : int
        Random seed (applied before each combo).

    Returns
    -------
    pd.DataFrame  –  one row per combo, sorted by final_equity descending.
    """
    if param_grid is None:
        param_grid = {
            "hidden_size":   [32, 64, 128],
            "num_layers":    [2, 3],
            "learning_rate": [0.001, 0.0005, 0.0001],
            "dropout":       [0.2, 0.3, 0.4],
            "seq_len":       [20, 40, 60],
            "batch_size":    [128, 256],
        }

    bt_cfg = CONFIG["backtest"]
    tr_cfg = CONFIG["trading"]

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_combos = len(combos)

    print("\n" + f" GRID SEARCH – LSTM ({n_combos} combos, walk-forward) ".center(PRINT_WIDTH, "="))

    results = []
    for idx, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n{'─'*PRINT_WIDTH}")
        print(f"  [{idx}/{n_combos}] {label}")

        lstm_mod.set_seed(seed)
        saved = _patch_lstm_globals(params)

        t0 = time.time()
        try:
            res, trades = backtest_triple_barrier(
                df,
                backtest_window=bt_cfg["backtest_window"],
                predict_window=bt_cfg["predict_window"],
                initial_capital=tr_cfg["initial_capital"],
                lookahead=tr_cfg["lookahead"],
                atr_mult=tr_cfg["atr_mult"],
                threshold=tr_cfg["threshold"],
                position_size=tr_cfg["position_size"],
                model_type="lstm",
            )
            summary = res.summary
        except Exception as e:
            print(f"  !! FAILED: {e}")
            summary = {"final_equity": 0.0, "trades": 0, "wins": 0,
                       "losses": 0, "hit_rate": 0.0, "max_drawdown": 0.0}
        finally:
            _restore_lstm_globals(saved)

        elapsed = time.time() - t0
        row = {
            **params,
            "final_equity": summary["final_equity"],
            "total_return_pct": (summary["final_equity"] - tr_cfg["initial_capital"]) / tr_cfg["initial_capital"],
            "trades": summary["trades"],
            "hit_rate": summary["hit_rate"],
            "max_drawdown": summary["max_drawdown"],
            "time_s": round(elapsed, 1),
        }
        results.append(row)
        print(f"  => equity=${row['final_equity']:,.2f}  "
              f"ret={row['total_return_pct']:+.2%}  "
              f"trades={row['trades']}  hit={row['hit_rate']:.2%}  "
              f"dd={row['max_drawdown']:.2%}  ({elapsed:.0f}s)")

    results_df = pd.DataFrame(results).sort_values("final_equity", ascending=False).reset_index(drop=True)
    print("\n" + " LSTM GRID SEARCH RESULTS (top 10) ".center(PRINT_WIDTH, "="))
    print(results_df.head(10).to_string(index=False))
    return results_df


def grid_search_mlp(
    df: pd.DataFrame,
    param_grid: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward grid search for the MLP classifier.

    Each combo runs the full ``backtest_triple_barrier`` with model_type="mlp".
    Results are sorted by **final_equity** (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Full enhanced dataset (with features & target already computed).
    param_grid : dict, optional
        Keys: hidden_sizes, learning_rate, dropout, batch_size.
        Values are lists of candidates.  A sensible small default is provided.
    seed : int
        Random seed (applied before each combo).

    Returns
    -------
    pd.DataFrame  –  one row per combo, sorted by final_equity descending.
    """
    if param_grid is None:
        param_grid = {
            "hidden_sizes":  [[64, 32, 64], [128, 64, 128], [256, 128, 256],
                              [128, 64, 32, 64, 128], [256, 128, 64, 128, 256]],
            "learning_rate": [0.001, 0.0005, 0.0001],
            "dropout":       [0.2, 0.3, 0.4],
            "batch_size":    [64, 128, 256],
        }

    bt_cfg = CONFIG["backtest"]
    tr_cfg = CONFIG["trading"]

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_combos = len(combos)

    print("\n" + f" GRID SEARCH – MLP ({n_combos} combos, walk-forward) ".center(PRINT_WIDTH, "="))

    results = []
    for idx, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n{'─'*PRINT_WIDTH}")
        print(f"  [{idx}/{n_combos}] {label}")

        mlp_mod.set_seed(seed)
        saved = _patch_mlp_globals(params)

        t0 = time.time()
        try:
            res, trades = backtest_triple_barrier(
                df,
                backtest_window=bt_cfg["backtest_window"],
                predict_window=bt_cfg["predict_window"],
                initial_capital=tr_cfg["initial_capital"],
                lookahead=tr_cfg["lookahead"],
                atr_mult=tr_cfg["atr_mult"],
                threshold=tr_cfg["threshold"],
                position_size=tr_cfg["position_size"],
                model_type="mlp",
            )
            summary = res.summary
        except Exception as e:
            print(f"  !! FAILED: {e}")
            summary = {"final_equity": 0.0, "trades": 0, "wins": 0,
                       "losses": 0, "hit_rate": 0.0, "max_drawdown": 0.0}
        finally:
            _restore_mlp_globals(saved)

        elapsed = time.time() - t0
        row = {
            **params,
            "final_equity": summary["final_equity"],
            "total_return_pct": (summary["final_equity"] - tr_cfg["initial_capital"]) / tr_cfg["initial_capital"],
            "trades": summary["trades"],
            "hit_rate": summary["hit_rate"],
            "max_drawdown": summary["max_drawdown"],
            "time_s": round(elapsed, 1),
        }
        results.append(row)
        print(f"  => equity=${row['final_equity']:,.2f}  "
              f"ret={row['total_return_pct']:+.2%}  "
              f"trades={row['trades']}  hit={row['hit_rate']:.2%}  "
              f"dd={row['max_drawdown']:.2%}  ({elapsed:.0f}s)")

    results_df = pd.DataFrame(results).sort_values("final_equity", ascending=False).reset_index(drop=True)
    print("\n" + " MLP GRID SEARCH RESULTS (top 10) ".center(PRINT_WIDTH, "="))
    print(results_df.head(10).to_string(index=False))
    return results_df


# ═══════════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataset_utils import dataset_utils, feature_engineering

    path_list = [Path("datasets/raw/ETHUSD_D1.csv")]
    if not dataset_utils.validate_dataset(path_list):
        print("Dataset validation failed.")
        sys.exit(1)

    lookahead = CONFIG["trading"]["lookahead"]
    atr_mult = CONFIG["trading"]["atr_mult"]
    path_list_final = feature_engineering.calculate_features(
        path_list, lookahead=lookahead, atr_mult=atr_mult,
    )
    df = pd.read_csv(path_list_final[0])

    print("\n\x1b[36m>>> Running LSTM grid search (walk-forward) …\x1b[0m")
    lstm_results = grid_search_lstm(df)

    print("\n\x1b[36m>>> Running MLP grid search (walk-forward) …\x1b[0m")
    mlp_results = grid_search_mlp(df)

    # Save full results to CSV for later analysis
    lstm_results.to_csv("grid_search_lstm_results.csv", index=False)
    mlp_results.to_csv("grid_search_mlp_results.csv", index=False)
    print("\n\x1b[32;1m>>> Results saved to grid_search_lstm_results.csv & grid_search_mlp_results.csv\x1b[0m")
