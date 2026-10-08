from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
from ollama import Client
from sentence_transformers import SentenceTransformer, util

from activation_drift.utils import infer_targets_from_condition, ollama_chat_with_retries, save_pickle


def compute_D(values):
    values_arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(values_arr))


def compute_TV(values):
    values_arr = np.asarray(values, dtype=np.float64)
    # return float(values_arr[-1] - values_arr[0])  ### ND ###
    return float(np.sum(np.abs(np.diff(values_arr))) / (len(values_arr) - 1)) ### TV ###


def compute_alignment_metrics(alignment_by_turn_df, use_delta_with_baseline=True):
    A_t_column = "delta_A_t" if use_delta_with_baseline else 'A_t'
    run_key_col = 'seed'
    base_group_cols = [
        'conversation_id', 'scenario_key', 'condition', 'scenario_value_a', 'scenario_value_b',
        'model_target_value', 'user_target_value',
    ]
    group_cols = base_group_cols + [run_key_col]

    seed_metric_rows = []
    for key, g in alignment_by_turn_df.groupby(group_cols, dropna=False):
        g = g.sort_values('turn')
        raw_series = g[A_t_column].to_numpy(dtype=np.float64)
        row = {col: value for col, value in zip(group_cols, key)}
        row['n_turns'] = int(len(g))
        row['D'] = compute_D(raw_series)
        row['TV'] = compute_TV(raw_series)
        seed_metric_rows.append(row)

    metric_df = pd.DataFrame(seed_metric_rows)
    metric_df['n_turns'] = metric_df['n_turns'].astype(int)
    metric_df = (
        metric_df.groupby(base_group_cols, dropna=False)
        .agg(
            n_seeds=(run_key_col, 'nunique'),
            n_turns=('n_turns', 'mean'),
            D_mean=('D', 'mean'),
            D_std=('D', 'std'),
            TV_mean=('TV', 'mean'),
            TV_std=('TV', 'std'),
        )
        .reset_index()
    )
    metric_df['n_turns'] = metric_df['n_turns'].astype(int)
    metric_df['n_seeds'] = metric_df['n_seeds'].astype(int)

    return metric_df


QUALITY_BASE_COLS = [
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


def _extract_run_metadata(rec: dict[str, Any]) -> dict[str, Any]:
    conv_id = str(rec.get("conversation_id", "")).strip()
    scenario_key = str(rec.get("scenario_key", conv_id)).strip() or conv_id
    scenario_value_a = str(rec.get("scenario_value_a", "")).strip()
    scenario_value_b = str(rec.get("scenario_value_b", "")).strip()
    steering_mode = str(rec.get("steering_mode", "")).strip()
    user_mode = str(rec.get("user_mode", "")).strip()
    repeat_index = int(rec.get("repeat_index", 0))
    seed = int(rec.get("seed", 0))

    condition_payload = rec.get("condition") or {}
    if isinstance(condition_payload, dict):
        condition_name = str(condition_payload.get("name", "")).strip()
        model_target_value = str(condition_payload.get("model_target_value", "")).strip()
        user_target_value = str(condition_payload.get("user_target_value", "")).strip()
    else:
        condition_name = str(condition_payload).strip()
        model_target_value = str(rec.get("model_target_value", "")).strip()
        user_target_value = str(rec.get("user_target_value", "")).strip()

    if (not model_target_value) and (not user_target_value):
        model_target_value, user_target_value = infer_targets_from_condition(
            condition=condition_name,
            value_a=scenario_value_a,
            value_b=scenario_value_b,
        )

    return {
        "conversation_id": conv_id,
        "scenario_key": scenario_key,
        "scenario_value_a": scenario_value_a,
        "scenario_value_b": scenario_value_b,
        "steering_mode": steering_mode,
        "condition": condition_name,
        "model_target_value": model_target_value,
        "user_target_value": user_target_value,
        "user_mode": user_mode,
        "repeat_index": repeat_index,
        "seed": seed,
    }


def _aggregate_turn_metric(metric_by_turn_df: pd.DataFrame, metric_col: str, metric_name: str):
    if metric_by_turn_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    run_group_cols = QUALITY_BASE_COLS + ["repeat_index", "seed"]
    per_run_df = (
        metric_by_turn_df
        .groupby(run_group_cols, as_index=False, dropna=False)
        .agg(
            n_turns=("turn", "size"),
            **{
                f"{metric_name}_mean": (metric_col, "mean"),
                f"{metric_name}_std_turn": (metric_col, "std"),
            },
        )
        .sort_values(["conversation_id", "condition", "seed"])
        .reset_index(drop=True)
    )
    per_run_df[f"{metric_name}_std_turn"] = per_run_df[f"{metric_name}_std_turn"].fillna(0.0)

    repeat_stats_df = (
        per_run_df
        .groupby(QUALITY_BASE_COLS, as_index=False, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_turns=("n_turns", "mean"),
            **{
                f"{metric_name}_mean": (f"{metric_name}_mean", "mean"),
                f"{metric_name}_std": (f"{metric_name}_mean", "std"),
            },
        )
        .sort_values(["conversation_id", "condition"])
        .reset_index(drop=True)
    )
    repeat_stats_df["n_seeds"] = repeat_stats_df["n_seeds"].astype(int)
    repeat_stats_df["n_turns"] = repeat_stats_df["n_turns"].astype(int)
    repeat_stats_df[f"{metric_name}_std"] = repeat_stats_df[f"{metric_name}_std"].fillna(0.0)

    scenario_setting_cols = [
        "scenario_key",
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
    ]
    scenario_stats_df = (
        repeat_stats_df
        .groupby(scenario_setting_cols, as_index=False, dropna=False)
        .agg(
            **{f"{metric_name}_mean": (f"{metric_name}_mean", "mean")},
            scenario_conversations=("conversation_id", "nunique"),
        )
        .sort_values(["scenario_value_a", "scenario_value_b", "condition", "scenario_key"])
        .reset_index(drop=True)
    )
    scenario_stats_df["scenario_conversations"] = scenario_stats_df["scenario_conversations"].astype(int)

    tradeoff_group_cols = [
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
    ]
    tradeoff_stats_df = (
        scenario_stats_df
        .groupby(tradeoff_group_cols, as_index=False, dropna=False)
        .agg(
            n_scenarios=("scenario_key", "nunique"),
            **{
                f"{metric_name}_mean": (f"{metric_name}_mean", "mean"),
                f"{metric_name}_std": (f"{metric_name}_mean", "std"),
            },
        )
        .sort_values(["scenario_value_a", "scenario_value_b", "condition"])
        .reset_index(drop=True)
    )
    tradeoff_stats_df["n_scenarios"] = tradeoff_stats_df["n_scenarios"].astype(int)
    tradeoff_stats_df[f"{metric_name}_std"] = tradeoff_stats_df[f"{metric_name}_std"].fillna(0.0)

    return per_run_df, repeat_stats_df, tradeoff_stats_df


def _make_metric_summary_df(tradeoff_stats_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if tradeoff_stats_df.empty:
        return tradeoff_stats_df.copy()

    summary_cols = [
        "scenario_value_a",
        "scenario_value_b",
        "steering_mode",
        "condition",
        "model_target_value",
        "user_target_value",
        "user_mode",
        "n_scenarios",
        f"{metric_name}_mean",
        f"{metric_name}_std",
    ]
    summary_df = tradeoff_stats_df[summary_cols].copy()
    float_cols = summary_df.select_dtypes(include=[np.floating]).columns
    if len(float_cols) > 0:
        summary_df[float_cols] = summary_df[float_cols].round(4)
    return summary_df


def _coherence_default_config_path() -> Path:
    return Path(__file__).with_name("eval_coherence.yaml")


def _load_coherence_prompt_templates(config_path: str | Path | None = None) -> tuple[str, str]:
    cfg_path = Path(config_path) if config_path else _coherence_default_config_path()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    judge_system_prompt = str(cfg.get("judge_system_prompt", "")).strip()
    judge_user_prompt_template = str(cfg.get("judge_user_prompt_template", "")).strip()

    if not judge_system_prompt or not judge_user_prompt_template:
        raise ValueError(
            "Coherence config must define non-empty keys: "
            "judge_system_prompt and judge_user_prompt_template"
        )

    return judge_system_prompt, judge_user_prompt_template


def _format_conversation_prefix_for_judge(
    user_turns: list[str],
    assistant_responses: list[str],
    upto_turn: int,
) -> str:
    lines: list[str] = []
    for t in range(upto_turn + 1):
        user_text = str(user_turns[t]).strip() if t < len(user_turns) else ""
        assistant_text = str(assistant_responses[t]).strip() if t < len(assistant_responses) else ""
        if user_text:
            lines.append(f"User (turn {t + 1}): {user_text}")
        if assistant_text:
            lines.append(f"Assistant (turn {t + 1}): {assistant_text}")
    return "\n".join(lines).strip()


def _extract_coherence_score(raw_output: str) -> int | None:
    text = str(raw_output or "").strip()
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        raise ValueError(f"Judge did not return an integer score: {text!r}")
    score = int(m.group(1))
    if score < 0 or score > 100:
        raise ValueError(f"Judge score out of range [0, 100]: {score}")
    return score


def _split_sentences(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _strip_reasoning_trail(text: str) -> tuple[str, bool]:
    think_block_pattern = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
    think_open_pattern = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
    cleaned_text = think_block_pattern.sub("", str(text or ""))
    unmatched_open = think_open_pattern.search(cleaned_text)
    if unmatched_open:
        return cleaned_text[:unmatched_open.start()].strip(), True
    return cleaned_text.strip(), False


class SentenceEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = str(model_name)
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


def evaluate_coherence_for_turn(
    client: Client,
    judge_model: str,
    judge_system_prompt: str,
    judge_user_prompt_template: str,
    user_turns: list[str],
    assistant_responses: list[str],
    turn: int,
    scenario_key: str,
    condition_name: str,
    temperature: float,
) -> float | None:
    conversation_prefix = _format_conversation_prefix_for_judge(
        user_turns=user_turns,
        assistant_responses=assistant_responses,
        upto_turn=turn,
    )

    judge_user_prompt = judge_user_prompt_template.format(
        conversation_prefix=conversation_prefix,
        turn_number=int(turn + 1),
        max_turns=int(len(assistant_responses)),
        scenario_key=scenario_key,
        condition=condition_name,
    )

    judge_output = ollama_chat_with_retries(
        client=client,
        model=judge_model,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt},
        ],
        options={"temperature": float(max(temperature, 0.0)), "num_predict": 8192},
        max_retries=6,
        base_delay=1.0,
    )
    score = _extract_coherence_score(judge_output)
    return float(score) if score is not None else None


def evaluate_repetition_for_turn(
    assistant_text: str,
    prev_assistant_texts: list[str],
    embedder: SentenceEmbedder,
    threshold: float = 0.8,
) -> dict[str, Any]:
    cur_sents = _split_sentences(assistant_text)
    prev_sents: list[str] = []
    for prev_text in prev_assistant_texts:
        prev_sents.extend(_split_sentences(prev_text))

    if not cur_sents:
        return {
            "assistant_sentence_count": 0,
            "reused_sentence_count": 0,
            "repetition_rate": 0.0,
            "sentence_nn_sim_mean": 0.0,
            "sentence_nn_sim_max": 0.0,
            "repetition_threshold": float(threshold),
        }
    if not prev_sents:
        return {
            "assistant_sentence_count": len(cur_sents),
            "reused_sentence_count": 0,
            "repetition_rate": 0.0,
            "sentence_nn_sim_mean": 0.0,
            "sentence_nn_sim_max": 0.0,
            "repetition_threshold": float(threshold),
        }

    prev_emb = embedder.encode(prev_sents)
    cur_emb = embedder.encode(cur_sents)
    sims = util.cos_sim(cur_emb, prev_emb)
    nn_sim, _ = torch.max(sims, dim=1)
    reused = nn_sim >= float(threshold)

    return {
        "assistant_sentence_count": len(cur_sents),
        "reused_sentence_count": int(reused.sum().item()),
        "repetition_rate": float(reused.float().mean().item()) if reused.numel() else 0.0,
        "sentence_nn_sim_mean": float(nn_sim.mean().item()) if nn_sim.numel() else 0.0,
        "sentence_nn_sim_max": float(nn_sim.max().item()) if nn_sim.numel() else 0.0,
        "repetition_threshold": float(threshold),
    }


def evaluate_conversation_quality(
    run_records,
    measure_coherence: bool = True,
    measure_repetition: bool = False,
    judge_model: str = "gpt-oss:120b-cloud",
    max_turns: int = 10,
    config_path: str | Path | None = None,
    temperature: float = 0.0,
    repetition_threshold: float = 0.8,
    repetition_model_name: str = "all-mpnet-base-v2",
    dump_file: str | Path | None = None,
    remove_reasoning_trail: bool = False,
):
    if not measure_coherence and not measure_repetition:
        raise ValueError("At least one evaluation must be enabled.")

    results = {}
    details = {}
    if measure_coherence:
        judge_system_prompt, judge_user_prompt_template = _load_coherence_prompt_templates(config_path=config_path)
        client = Client(
            host="https://ollama.com",
            headers={"Authorization": "Bearer " + os.environ["OLLAMA_API_KEY"]},
        )

    repetition_embedder = SentenceEmbedder(model_name=repetition_model_name) if measure_repetition else None

    coherence_turn_rows = []
    repetition_turn_rows = []
    truncated_reasoning_responses = 0
    empty_after_stripping_turns = 0
    blank_judge_responses = 0
    for rec in tqdm(run_records):
        metadata = _extract_run_metadata(rec)

        user_turns = list(rec.get("turns") or [])
        assistant_responses = list(rec.get("responses") or [])
        if remove_reasoning_trail:
            cleaned_responses = []
            for response in assistant_responses:
                cleaned_response, was_truncated = _strip_reasoning_trail(response)
                cleaned_responses.append(cleaned_response)
                truncated_reasoning_responses += int(was_truncated)
            assistant_responses = cleaned_responses
        n_available_turns = min(len(user_turns), len(assistant_responses))
        n_eval_turns = min(int(max_turns), n_available_turns)

        for turn in range(n_eval_turns):
            if remove_reasoning_trail and not assistant_responses[turn].strip():
                empty_after_stripping_turns += 1
                print(
                    f"Skipping turn {turn + 1} for {metadata['conversation_id']}: "
                    f"empty after reasoning removal (count={empty_after_stripping_turns})"
                )
                continue

            base_row = dict(metadata)
            base_row["turn"] = int(turn)

            if measure_coherence:
                try:
                    score = evaluate_coherence_for_turn(
                        client=client,
                        judge_model=judge_model,
                        judge_system_prompt=judge_system_prompt,
                        judge_user_prompt_template=judge_user_prompt_template,
                        user_turns=user_turns,
                        assistant_responses=assistant_responses,
                        turn=turn,
                        scenario_key=metadata["scenario_key"],
                        condition_name=metadata["condition"],
                        temperature=temperature,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Invalid coherence judge output "
                        f"(conversation_id={metadata['conversation_id']}, condition={metadata['condition']}, "
                        f"seed={metadata['seed']}, turn={turn}, judge_model={judge_model})"
                    ) from exc
                if score is None:
                    blank_judge_responses += 1
                    print(
                        f"WARNING: Skipping coherence score for {metadata['conversation_id']} "
                        f"turn {turn + 1}: blank judge response (count={blank_judge_responses})"
                    )
                else:
                    coherence_turn_rows.append({**base_row, "coherence_score": float(score)})

            if measure_repetition:
                repetition_row = evaluate_repetition_for_turn(
                    assistant_text=assistant_responses[turn],
                    prev_assistant_texts=assistant_responses[:turn],
                    embedder=repetition_embedder,
                    threshold=repetition_threshold,
                )
                repetition_turn_rows.append({**base_row, **repetition_row})

    if remove_reasoning_trail:
        print(f"Skipped empty-after-stripping turns: {empty_after_stripping_turns}")
    if measure_coherence:
        print(f"Skipped blank judge responses: {blank_judge_responses}")

    if truncated_reasoning_responses:
        warnings.warn(
            f"Removed unterminated <think> content from {truncated_reasoning_responses} response(s); "
            "those turns may have no evaluable answer because generation ended during reasoning.",
            RuntimeWarning,
            stacklevel=2,
        )

    if measure_coherence:
        coherence_by_turn_df = pd.DataFrame(coherence_turn_rows)
        coherence_per_run_df, coherence_repeat_stats_df, coherence_tradeoff_stats_df = _aggregate_turn_metric(
            coherence_by_turn_df,
            metric_col="coherence_score",
            metric_name="coherence",
        )
        results["coherence"] = _make_metric_summary_df(coherence_tradeoff_stats_df, metric_name="coherence")
        details["coherence"] = {
            "by_turn_df": coherence_by_turn_df,
            "per_run_df": coherence_per_run_df,
            "repeat_stats_df": coherence_repeat_stats_df,
            "tradeoff_stats_df": coherence_tradeoff_stats_df,
            "summary_df": results["coherence"],
        }

    if measure_repetition:
        repetition_by_turn_df = pd.DataFrame(repetition_turn_rows)
        repetition_per_run_df, repetition_repeat_stats_df, repetition_tradeoff_stats_df = _aggregate_turn_metric(
            repetition_by_turn_df,
            metric_col="repetition_rate",
            metric_name="repetition",
        )
        results["repetition"] = _make_metric_summary_df(repetition_tradeoff_stats_df, metric_name="repetition")
        details["repetition"] = {
            "by_turn_df": repetition_by_turn_df,
            "per_run_df": repetition_per_run_df,
            "repeat_stats_df": repetition_repeat_stats_df,
            "tradeoff_stats_df": repetition_tradeoff_stats_df,
            "summary_df": results["repetition"],
        }

    if dump_file:
        dump_path = Path(dump_file)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                "measure_coherence": bool(measure_coherence),
                "measure_repetition": bool(measure_repetition),
                "judge_model": judge_model,
                "max_turns": int(max_turns),
                "config_path": str(config_path) if config_path is not None else None,
                "temperature": float(temperature),
                "repetition_threshold": float(repetition_threshold),
                "repetition_model_name": repetition_model_name,
                "remove_reasoning_trail": bool(remove_reasoning_trail),
                "num_run_records": int(len(run_records)),
            },
            "summary": results,
            "details": details,
        }
        save_pickle(payload, dump_path)

    return results
