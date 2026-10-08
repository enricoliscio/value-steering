from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from activation_drift.utils import infer_targets_from_condition


def choose_latest_run(artifacts_root: Path):
    candidates = sorted(
        [p for p in artifacts_root.glob("probe_experiment_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError("No artifacts/probe_experiment_* directory found.")

    RESULT_DIR = candidates[0]
    print(f"Using result directory: {RESULT_DIR}")
    return RESULT_DIR


def load_raw_results(RESULT_DIR: str, normalization_json=""):
    probe_scores_by_turn_path = RESULT_DIR / "probe_scores_by_turn.csv"
    runs_jsonl_path = RESULT_DIR / "runs.jsonl"
    normalization_path = Path(normalization_json)

    probe_scores_by_turn_df = pd.read_csv(probe_scores_by_turn_path) if probe_scores_by_turn_path.exists() else pd.DataFrame()
    if normalization_path.exists():
        normalization_stats = json.loads(normalization_path.read_text(encoding="utf-8"))
        if "per_value" in normalization_stats.keys():
            normalization_stats = normalization_stats["per_value"]
    else:
        raise ValueError("Could not find calibration file. make sure it's present in the results directory.")

    run_records = []
    if runs_jsonl_path.exists():
        with runs_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    run_records.append(json.loads(line))

    return probe_scores_by_turn_df, normalization_stats, run_records


def load_config(RESULT_DIR: str):
    config_path = RESULT_DIR / "experiment_config.json"
    partial_config_path = RESULT_DIR / "experiment_config.partial.json"

    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    elif partial_config_path.exists():
        return json.loads(partial_config_path.read_text(encoding="utf-8"))
    else:
        return ValueError("Could not find a config file.")


def get_tradeoffs_per_conversation(run_records):
    tradeoff_by_conv = {}
    run_meta_by_key = {}
    for rec in run_records:
        conv_id = str(rec.get("conversation_id", "")).strip()

        condition = rec["condition"]
        condition_name = str(condition.get("name", "")).strip()
        model_target = str(condition.get("model_target_value", "")).strip()
        user_target = str(condition.get("user_target_value", "")).strip()

        sv_a = str(rec.get("scenario_value_a", "")).strip()
        sv_b = str(rec.get("scenario_value_b", "")).strip()
        tradeoff_by_conv[conv_id] = (sv_a, sv_b)

        if condition_name != "baseline" and (not model_target and not user_target):
            model_target, user_target = infer_targets_from_condition(
                condition_name,
                sv_a,
                sv_b
            )

        repeat_index = int(rec.get("repeat_index", 0))
        run_meta_by_key[(conv_id, condition_name, repeat_index)] = {
            "model_target_value": model_target,
            "user_target_value": user_target,
        }

    return tradeoff_by_conv, run_meta_by_key


def _norm_value_name(x: str) -> str:
    return str(x).strip().upper().replace("_", "-")


def _normalize_probe_logit(row: pd.Series, normalization_stats) -> float:
    value_name = str(row.get("value_name", "")).strip()
    raw = float(row.get("probe_logit", float("nan")))
    stats = normalization_stats.get(value_name)
    if not stats:
        return raw
    mu = float(stats.get("mu", 0.0))
    sigma = float(stats.get("sigma", 0.0))
    if sigma == 0.0:
        return raw
    return (raw - mu) / sigma


def aggregate_alignment_df(alignment_by_turn_df, A_t_column='A_t'):
    combo_cols = [
        "conversation_id",
        "scenario_key",
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
        "turn",
    ]
    alignment_repeat_stats_df = (
        alignment_by_turn_df
        .groupby(combo_cols, as_index=False, dropna=False)
        .agg(
            A_t_mean=(A_t_column, "mean"),
            A_t_std=(A_t_column, "std"),
            n_seeds=(A_t_column, "size"),
        )
        .sort_values(["conversation_id", "condition", "turn"])
        .reset_index(drop=True)
    )

    return alignment_repeat_stats_df


def correct_with_baseline(alignment_by_turn_df, alignment_repeat_stats_df):
    baseline_condition='baseline'
    baseline_df = alignment_repeat_stats_df[alignment_repeat_stats_df['condition'].astype(str).str.lower() == baseline_condition].copy()
    baseline_by_scenario_turn = (
            baseline_df.groupby(['scenario_key', 'turn'], as_index=False)
            .agg(baseline_A_t_mean=('A_t_mean', 'mean'))
        )
    baseline_corrected_df = (
            alignment_by_turn_df[
                ~alignment_by_turn_df['condition'].astype(str).str.lower().eq(baseline_condition)
            ]
            .merge(baseline_by_scenario_turn, on=['scenario_key', 'turn'], how='left')
            .copy()
        )

    baseline_corrected_df['delta_A_t'] = baseline_corrected_df['A_t'] - baseline_corrected_df['baseline_A_t_mean']
    baseline_corrected_df = baseline_corrected_df.sort_values(['scenario_key', 'condition', 'turn']).reset_index(drop=True)

    baseline_corrected_aggregated_df = aggregate_alignment_df(baseline_corrected_df, A_t_column='delta_A_t')

    return baseline_corrected_df, baseline_corrected_aggregated_df


def make_alignment_dfs(probe_scores_by_turn_df, run_meta_by_key, normalization_stats, use_delta_with_baseline=True):
    key_cols = [
        "conversation_id",
        "scenario_key",
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
        "repeat_index",
        "seed",
        "turn",
    ]
    
    df_probe = probe_scores_by_turn_df.copy()

    # Ensure target columns can hold strings before any backfill assignments.
    for col in ["model_target_value", "user_target_value"]:
        if col not in df_probe.columns:
            df_probe[col] = pd.Series([None] * len(df_probe), dtype=object)
        else:
            df_probe[col] = df_probe[col].astype(object)

    # Fill missing model/user targets from runs metadata first, then condition-name inference.
    probe_missing = (
        df_probe["model_target_value"].isna()
        | (df_probe["model_target_value"].astype(str).str.strip() == "")
        | df_probe["user_target_value"].isna()
        | (df_probe["user_target_value"].astype(str).str.strip() == "")
    )
    for i in df_probe.index[probe_missing]:
        conv_id = str(df_probe.at[i, "conversation_id"])
        condition_name = str(df_probe.at[i, "condition"])
        repeat_idx = int(pd.to_numeric(df_probe.at[i, "repeat_index"], errors="coerce") if pd.notna(df_probe.at[i, "repeat_index"]) else 0)
        meta = run_meta_by_key.get((conv_id, condition_name, repeat_idx))
        if meta:
            if not str(df_probe.at[i, "model_target_value"] if pd.notna(df_probe.at[i, "model_target_value"]) else "").strip():
                df_probe.at[i, "model_target_value"] = meta.get("model_target_value", "")
            if not str(df_probe.at[i, "user_target_value"] if pd.notna(df_probe.at[i, "user_target_value"]) else "").strip():
                df_probe.at[i, "user_target_value"] = meta.get("user_target_value", "")

    infer_vals = df_probe.apply(
        lambda r: infer_targets_from_condition(
            r.get("condition", ""),
            r.get("scenario_value_a", ""),
            r.get("scenario_value_b", "")        
        ),
        axis=1,
        result_type="expand",
    )
    infer_vals.columns = ["_model_target_infer", "_user_target_infer"]
    for col, infer_col in [
        ("model_target_value", "_model_target_infer"),
        ("user_target_value", "_user_target_infer"),
    ]:
        missing_mask = df_probe[col].isna() | (df_probe[col].astype(str).str.strip() == "")
        df_probe.loc[missing_mask, col] = infer_vals.loc[missing_mask, infer_col].to_numpy()

    # Prevent groupby from dropping rows because of NaN in key columns.
    for col in key_cols:
        if col not in df_probe.columns:
            df_probe[col] = ""
    for col in [
        "conversation_id",
        "scenario_key",
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
    ]:
        df_probe[col] = df_probe[col].fillna("<missing>").astype(str)
    for col in ["repeat_index", "seed", "turn"]:
        df_probe[col] = pd.to_numeric(df_probe[col], errors="coerce").fillna(-1).astype(int)

    df_probe["value_name_norm"] = df_probe["value_name"].map(_norm_value_name)
    df_probe["scenario_value_a_norm"] = df_probe["scenario_value_a"].map(_norm_value_name)
    df_probe["scenario_value_b_norm"] = df_probe["scenario_value_b"].map(_norm_value_name)

    df_probe["probe_logit_norm"] = df_probe.apply(_normalize_probe_logit, normalization_stats=normalization_stats, axis=1)

    a_rows = df_probe[df_probe["value_name_norm"] == df_probe["scenario_value_a_norm"]][key_cols + ["probe_logit", "probe_logit_norm"]].rename(
        columns={"probe_logit": "logit_a_raw", "probe_logit_norm": "logit_a"}
    )
    b_rows = df_probe[df_probe["value_name_norm"] == df_probe["scenario_value_b_norm"]][key_cols + ["probe_logit", "probe_logit_norm"]].rename(
        columns={"probe_logit": "logit_b_raw", "probe_logit_norm": "logit_b"}
    )

    alignment_by_turn_df = a_rows.merge(b_rows, on=key_cols, how="inner")
    alignment_by_turn_df["A_t"] = alignment_by_turn_df["logit_a"] - alignment_by_turn_df["logit_b"]
    alignment_by_turn_df["A_t_raw"] = alignment_by_turn_df["logit_a_raw"] - alignment_by_turn_df["logit_b_raw"]
    alignment_by_turn_df = alignment_by_turn_df.sort_values(["conversation_id", "condition", "repeat_index", "turn"]).reset_index(drop=True)

    alignment_repeat_stats_df = aggregate_alignment_df(alignment_by_turn_df, A_t_column="A_t")

    if use_delta_with_baseline:
        alignment_by_turn_df, alignment_repeat_stats_df = correct_with_baseline(alignment_by_turn_df, alignment_repeat_stats_df)

    return alignment_by_turn_df, alignment_repeat_stats_df


def build_turns_df(run_records, alignment_repeat_stats_df):
    text_lookup = {}
    for rec in run_records:
        conv_id = rec.get("conversation_id")
        condition_name = (rec.get("condition") or {}).get("name")
        repeat_index = int(rec.get("repeat_index", 0))
        turns = rec.get("turns", [])
        responses = rec.get("responses", [])
        max_len = max(len(turns), len(responses))
        for t in range(max_len):
            key = (conv_id, condition_name, t)
            payload = {
                "repeat_index": repeat_index,
                "user_turn": turns[t] if t < len(turns) else None,
                "assistant_response": responses[t] if t < len(responses) else None,
            }
            if key not in text_lookup or repeat_index < text_lookup[key]["repeat_index"]:
                text_lookup[key] = payload

    rows = []
    for _, row in alignment_repeat_stats_df.iterrows():
        key = (row["conversation_id"], row["condition"], int(row["turn"]))
        text_pair = text_lookup.get(
            key,
            {
                "repeat_index": None,
                "user_turn": None,
                "assistant_response": None,
            },
        )
        rows.append({
            "conversation_id": row["conversation_id"],
            "scenario_key": row["scenario_key"],
            "scenario_value_a": row["scenario_value_a"],
            "scenario_value_b": row["scenario_value_b"],
            "condition": row["condition"],
            "model_target_value": row["model_target_value"],
            "user_target_value": row["user_target_value"],
            "turn": int(row["turn"]),
            "n_seeds": int(row["n_seeds"]),
            "A_t_mean": float(row["A_t_mean"]),
            "A_t_std": float(row["A_t_std"]),
            "example_from_repeat": text_pair["repeat_index"],
            "example_user_turn": text_pair["user_turn"],
            "example_assistant_response": text_pair["assistant_response"],
        })

    turns_df = pd.DataFrame(rows).sort_values(["conversation_id", "condition", "turn"]).reset_index(drop=True)

    return turns_df


def parse_metrics_stats(metrics_per_run_df, tradeoff_by_conv, run_meta_by_key):
    if "repeat_index" not in metrics_per_run_df.columns:
        metrics_per_run_df["repeat_index"] = 0
    for col in ["model_target_value", "user_target_value"]:
        if col not in metrics_per_run_df.columns:
            metrics_per_run_df[col] = pd.Series([None] * len(metrics_per_run_df), dtype=object)
        else:
            metrics_per_run_df[col] = metrics_per_run_df[col].astype(object)

    if "scenario_value_a" not in metrics_per_run_df.columns:
        metrics_per_run_df["scenario_value_a"] = ""
    if "scenario_value_b" not in metrics_per_run_df.columns:
        metrics_per_run_df["scenario_value_b"] = ""
    conv_ids = metrics_per_run_df["conversation_id"].astype(str).fillna("")
    missing_a = metrics_per_run_df["scenario_value_a"].astype(str).str.strip() == ""
    missing_b = metrics_per_run_df["scenario_value_b"].astype(str).str.strip() == ""
    for i in metrics_per_run_df.index[missing_a | missing_b]:
        conv_id = conv_ids.loc[i]
        fallback = tradeoff_by_conv.get(conv_id)
        if fallback:
            if missing_a.loc[i]:
                metrics_per_run_df.at[i, "scenario_value_a"] = fallback[0]
            if missing_b.loc[i]:
                metrics_per_run_df.at[i, "scenario_value_b"] = fallback[1]

    metric_missing = (
        metrics_per_run_df["model_target_value"].isna()
        | (metrics_per_run_df["model_target_value"].astype(str).str.strip() == "")
        | metrics_per_run_df["user_target_value"].isna()
        | (metrics_per_run_df["user_target_value"].astype(str).str.strip() == "")
    )
    for i in metrics_per_run_df.index[metric_missing]:
        conv_id = str(metrics_per_run_df.at[i, "conversation_id"])
        condition_name = str(metrics_per_run_df.at[i, "condition"])
        repeat_idx = int(pd.to_numeric(metrics_per_run_df.at[i, "repeat_index"], errors="coerce") if pd.notna(metrics_per_run_df.at[i, "repeat_index"]) else 0)
        meta = run_meta_by_key.get((conv_id, condition_name, repeat_idx))
        if meta:
            if not str(metrics_per_run_df.at[i, "model_target_value"] if pd.notna(metrics_per_run_df.at[i, "model_target_value"]) else "").strip():
                metrics_per_run_df.at[i, "model_target_value"] = meta.get("model_target_value", "")
            if not str(metrics_per_run_df.at[i, "user_target_value"] if pd.notna(metrics_per_run_df.at[i, "user_target_value"]) else "").strip():
                metrics_per_run_df.at[i, "user_target_value"] = meta.get("user_target_value", "")

    infer_vals = metrics_per_run_df.apply(
        lambda r: infer_targets_from_condition(
            r.get("condition", ""),
            r.get("scenario_value_a", ""),
            r.get("scenario_value_b", "")
        ),
        axis=1,
        result_type="expand",
    )
    infer_vals.columns = ["_model_target_infer", "_user_target_infer"]
    for col, infer_col in [
        ("model_target_value", "_model_target_infer"),
        ("user_target_value", "_user_target_infer"),
    ]:
        missing_mask = metrics_per_run_df[col].isna() | (metrics_per_run_df[col].astype(str).str.strip() == "")
        metrics_per_run_df.loc[missing_mask, col] = infer_vals.loc[missing_mask, infer_col].to_numpy()

    metric_combo_cols = [
        "conversation_id",
        "scenario_key",
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
    ]

    for col in metric_combo_cols:
        if col not in metrics_per_run_df.columns:
            metrics_per_run_df[col] = ""
        metrics_per_run_df[col] = metrics_per_run_df[col].fillna("<missing>").astype(str)

    metrics_repeat_stats_df = (
        metrics_per_run_df
        .groupby(metric_combo_cols, as_index=False, dropna=False)
        .agg(
            n_runs=("D", "size"),
            D_mean=("D", "mean"),
            D_std=("D", "std"),
            TV_mean=("TV", "mean"),
            TV_std=("TV", "std"),
        )
        .sort_values(["conversation_id", "condition"])
        .reset_index(drop=True)
    )
    metrics_repeat_stats_df[["D_std", "TV_std"]] = metrics_repeat_stats_df[["D_std", "TV_std"]].fillna(0.0)

    return metrics_repeat_stats_df


def print_data_summary(config, run_records, probe_scores_by_turn_df, alignment_by_turn_df, alignment_repeat_stats_df):
    expected_n = int(config.get("num_seed_runs", 1)) if config else 1
    status = str(config.get("status", "complete")) if config else "unknown"
    completed_conversations = int(config.get("completed_conversations", 0)) if config else 0
    print(f"Loaded runs: {len(run_records)}")
    print(f"probe_scores_by_turn rows: {len(probe_scores_by_turn_df)}")
    print(f"alignment_by_turn rows (raw): {len(alignment_by_turn_df)}")
    print(f"alignment_repeat_stats rows: {len(alignment_repeat_stats_df)}")
    print(f"Expected repeated runs per combination (config): {expected_n}")
    print(f"Run status: {status}")
    if completed_conversations:
        print(f"Completed conversations reported: {completed_conversations}")

