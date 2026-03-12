# import pandas as pd

# # -- Ranking columns: all "higher is better" --------------------------
# # max_drawdown is negative, so higher (closer to 0) = better
# RANK_COLS = ["total_return_pct", "hit_rate", "max_drawdown", "RoverMDD"]
# TOP_N = 20  # how many to show in tier list


# def tier_list(df: pd.DataFrame, label: str, id_cols: list[str]) -> pd.DataFrame:
#     """Rank models by each column, average the ranks, sort."""
#     for col in RANK_COLS:
#         # rank 1 = best (highest value)
#         df[f"rank_{col}"] = df[col].rank(ascending=False, method="min")

#     rank_cols = [f"rank_{col}" for col in RANK_COLS]
#     df["avg_rank"] = df[rank_cols].mean(axis=1)
#     df = df.sort_values("avg_rank").reset_index(drop=True)
#     df.index += 1  # 1-based position
#     df.index.name = "pos"

#     # -- Print summary -------------------------------------------------
#     total_time = pd.to_timedelta(df["time_s"].sum(), unit="s")
#     print(f"\n{'='*70}")
#     print(f"  {label}  -  {len(df)} configs  |  Total time: {total_time}")
#     print(f"{'='*70}")

#     show_cols = id_cols + RANK_COLS + rank_cols + ["avg_rank"]
#     print(df[show_cols].head(TOP_N).to_string())
#     return df


# # -- MLP ---------------------------------------------------------------
# mlp = pd.read_csv("grid_search_mlp_results.csv")
# mlp_id = ["hidden_sizes", "learning_rate", "dropout", "batch_size"]
# mlp_ranked = tier_list(mlp, "MLP TIER LIST", mlp_id)

# # -- LSTM --------------------------------------------------------------
# lstm = pd.read_csv("grid_search_lstm_results.csv")
# lstm_id = ["hidden_size", "num_layers", "learning_rate", "dropout", "seq_len", "batch_size"]
# lstm_ranked = tier_list(lstm, "LSTM TIER LIST", lstm_id)

# # -- Cross-model comparison (normalize ranks to [0,1] so different 
# #    pool sizes are comparable) ----------------------------------------
# print(f"\n{'='*70}")
# print("  OVERALL TOP 10 (normalized rank across both models)")
# print(f"{'='*70}")

# for tag, df, id_cols in [("MLP", mlp_ranked, mlp_id), ("LSTM", lstm_ranked, lstm_id)]:
#     n = len(df)
#     for col in RANK_COLS:
#         df[f"nrank_{col}"] = df[f"rank_{col}"] / n
#     df["avg_nrank"] = df[[f"nrank_{col}" for col in RANK_COLS]].mean(axis=1)
#     df["model"] = tag

# combined = pd.concat([mlp_ranked, lstm_ranked], ignore_index=True)
# combined = combined.sort_values("avg_nrank").reset_index(drop=True)
# combined.index += 1
# combined.index.name = "pos"

# show = ["model"] + ["total_return_pct", "hit_rate", "max_drawdown", "RoverMDD", "avg_nrank"]
# print(combined[show].head(10).to_string())


import pandas as pd

df = pd.read_csv("machine_learning/grid_search/4_grid_search_lstm_results.csv")

# total execution time in hh:mm:ss
total_time = pd.to_timedelta(df["time_s"].sum(), unit="s")
print(f"Total execution time: {total_time}")

# max total return
print("\n==> TOP 10 COMBOS BY TOTAL RETURN <==\n")
print(df.sort_values("total_return_pct", ascending=False).head(5))

# max hit rate
print("\n==> TOP 10 COMBOS BY HIT RATE <==\n")
print(df.sort_values("hit_rate", ascending=False).head(5))

# max drawdown
print("\n==> TOP 10 COMBOS BY MIN MAXDRAWDOWN <==\n")
print(df.sort_values("max_drawdown", ascending=False).head(5))

# max RoverMDD
print("\n==> TOP 10 COMBOS BY ROVERMDD <==\n")
print(df.sort_values("RoverMDD", ascending=False).head(5))