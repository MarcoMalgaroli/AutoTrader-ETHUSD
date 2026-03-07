"""
Configuration utilities for lookahead-aware config resolution.

The project supports multiple lookahead horizons (e.g. 4 and 10).  Model and
trading hyper-parameters can differ between horizons.  This module resolves the
correct config section based on the **active** lookahead stored in
``config["trading"]["lookahead"]``.

Two config files
----------------
* ``config.default.json`` - immutable baseline (committed to VCS)
* ``config.json``         - user-editable copy (written by the dashboard)

Convention in config.json
-------------------------
* ``lstm_classifier_10`` / ``lstm_classifier_4`` - LSTM hyper-parameters
* ``mlp_10`` / ``mlp_4``                         - MLP hyper-parameters
* ``trading_10`` / ``trading_4``                  - per-horizon trading params
* ``trading``                                     - common trading params
  (lookahead selector, initial_capital, commission)

Usage::

    from config_utils import get_trading_config, get_model_config

    tr  = get_trading_config(CONFIG)            # merged trading dict
    cfg = get_model_config("lstm_classifier", CONFIG)  # model dict for active LA
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.default.json"


def load_config(path: str | Path | None = None) -> dict:
    """Load *config.json* and return the raw dict."""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r") as f:
        return json.load(f)


def load_default_config() -> dict:
    """Load *config.default.json* and return the raw dict."""
    with open(_DEFAULT_CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config: dict, path: str | Path | None = None) -> None:
    """Persist *config* dict to disk (``config.json`` by default)."""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "w") as f:
        json.dump(config, f, indent=4)


def reset_to_defaults(path: str | Path | None = None) -> dict:
    """Overwrite ``config.json`` with ``config.default.json`` and return it."""
    p = Path(path) if path else _CONFIG_PATH
    shutil.copy2(_DEFAULT_CONFIG_PATH, p)
    return load_config(p)


def get_trading_config(config: dict | None = None, lookahead: int | None = None) -> dict:
    """Return **merged** trading config for a given lookahead.

    Merges the base ``trading`` section with ``trading_{lookahead}`` overrides.
    If *lookahead* is ``None``, uses ``config["trading"]["lookahead"]``.
    """
    if config is None:
        config = load_config()
    if lookahead is None:
        lookahead = config["trading"]["lookahead"]
    base = config["trading"].copy()
    override = config.get(f"trading_{lookahead}", {})
    base.update(override)
    return base


def get_model_config(model_type: str, config: dict | None = None, lookahead: int | None = None) -> dict:
    """Return model config for a given lookahead.

    Looks for ``{model_type}_{lookahead}`` first, then falls back to
    ``{model_type}``.
    """
    if config is None:
        config = load_config()
    if lookahead is None:
        lookahead = config["trading"]["lookahead"]
    key = f"{model_type}_{lookahead}"
    if key in config:
        return config[key]
    return config.get(model_type, {})
