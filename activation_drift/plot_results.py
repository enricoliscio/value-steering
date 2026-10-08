import pandas as pd
import numpy as np
from html import escape
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, Output, VBox


def plot_D_TV(metric_df):
    plot_metrics = (
        metric_df
        .groupby("condition", as_index=False)
        .agg(
            D_mean=("D_mean", "mean"),
            D_std=("D_mean", "std"),
            TV_mean=("TV_mean", "mean"),
            TV_std=("TV_mean", "std"),
            conversations_per_condition=("conversation_id", "count"),
            seeds_per_conversation=("n_seeds", "mean"),
        )
        .sort_values("condition")
        .reset_index(drop=True)
    )
    plot_metrics[["D_std", "TV_std"]] = plot_metrics[["D_std", "TV_std"]].fillna(0.0)
    ylabel_d = "D"
    ylabel_tv = "TV"
    title_d = "Net Drift (D) by Condition"
    title_tv = "Total Variation (TV) by Condition"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = plot_metrics["condition"].tolist()

    axes[0].bar(order, plot_metrics["D_mean"], yerr=plot_metrics["D_std"], capsize=3)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title(title_d)
    axes[0].set_ylabel(ylabel_d)
    axes[0].tick_params(axis="x", rotation=90)

    axes[1].bar(order, plot_metrics["TV_mean"], yerr=plot_metrics["TV_std"], capsize=3)
    axes[1].set_title(title_tv)
    axes[1].set_ylabel(ylabel_tv)
    axes[1].tick_params(axis="x", rotation=90)

    fig.tight_layout()
    plt.show()

    return plot_metrics[["condition", "conversations_per_condition", "seeds_per_conversation", "D_mean", "D_std", "TV_mean", "TV_std"]]


def plot_specific_trajectory(
        alignment_repeat_stats_df,
        selected_conversation="tradeoff1_workplace_norm",
        selected_conditions=["both_va", "both_vb"],
        seed=None,
        alignment_by_turn_df=None,
    ):
    # With seed set, plot that seed's raw A_t from alignment_by_turn_df instead of the mean over seeds.
    if seed is not None:
        if alignment_by_turn_df is None:
            raise ValueError("alignment_by_turn_df is required when seed is provided")
        plot_df = alignment_by_turn_df[alignment_by_turn_df["seed"] == seed].copy()
        if plot_df.empty:
            raise ValueError(f"No rows found for seed={seed}")
        plot_df = plot_df.rename(columns={"A_t": "A_t_mean"})
        plot_df["A_t_std"] = 0.0
        plot_df["n_seeds"] = 1
    else:
        plot_df = alignment_repeat_stats_df.copy()
    plot_df = plot_df.dropna(subset=["A_t_mean", "turn"])

    plot_df = plot_df[plot_df["condition"].isin(selected_conditions)].copy()
    plot_df = plot_df[plot_df["conversation_id"] == selected_conversation].copy()

    value_pair = plot_df[["scenario_value_a", "scenario_value_b"]].drop_duplicates()
    if len(value_pair) >= 1:
        value_a = value_pair.iloc[0]["scenario_value_a"]
        value_b = value_pair.iloc[0]["scenario_value_b"]
    else:
        value_a = "value_a"
        value_b = "value_b"

    plt.figure(figsize=(10, 6))
    data = {}
    for cond, g in plot_df.groupby("condition"):
        g = g.sort_values("turn")
        x = g["turn"].to_numpy()
        y = g["A_t_mean"].to_numpy()
        if "A_t_std" in g.columns:
            s = g["A_t_std"].fillna(0.0).to_numpy()
        else:
            s = np.zeros(len(g), dtype=float)
        n_seeds = int(g["n_seeds"].iloc[0]) if len(g) else 0
        plt.plot(x, y, marker="o", label=f"{cond} (n_seeds={n_seeds})")
        plt.fill_between(x, y - s, y + s, alpha=0.2)
        data[cond] = {}
        data[cond]['A_t_mean'] = y
        data[cond]['A_t_std'] = s

    plt.axhline(0.0, color="black", linewidth=1)
    seed_label = f" | seed={seed}" if seed is not None else ""
    plt.title(f"A_t | conversation={selected_conversation}{seed_label}")
    plt.xlabel("Turn t")
    plt.ylabel(f"A_t (+: toward {value_a} | -: toward {value_b})")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()

    return data


def _plot_panel(ax, subset, value_a, value_b, toward_value, condition):
    if condition == "both":
        value_model = toward_value
        value_user = toward_value
    elif condition == "opposite":
        value_model = toward_value
        values = [value_a, value_b]
        values.remove(toward_value)
        value_user = values[0]
    elif condition == "only_user":
        value_model = ""
        value_user = toward_value
    else:
        return ValueError("Condition must be in [both, opposite, only user].")

    g_sel = subset[
        (subset["scenario_value_a"] == value_a)
        & (subset["scenario_value_b"] == value_b)
        & (subset["model_target_value"] == value_model)
        & (subset["user_target_value"] == value_user)
    ].copy()

    g_sel = g_sel.assign(
        combo_key=(
            g_sel["conversation_id"].astype(str) + "::" + g_sel["condition"].astype(str)
        )
    )
    g_stats = (
        g_sel.groupby("turn", as_index=False)
        .agg(
            A_t_mean=("A_t_mean", "mean"),
            A_t_std=("A_t_mean", "std"),
            n_scenarios=("combo_key", "nunique"),
        )
        .sort_values("turn")
    )
    g_stats["A_t_std"] = g_stats["A_t_std"].fillna(0.0).clip(lower=0.0)

    x = g_stats["turn"].to_numpy()
    y = g_stats["A_t_mean"].to_numpy()
    s = g_stats["A_t_std"].to_numpy()
    n_scenarios = int(round(g_stats["n_scenarios"].mean()))

    ax.plot(x, y, marker="o", label=f"mean ΔA_t (n_scenarios={n_scenarios})")
    ax.fill_between(x, y - s, y + s, alpha=0.2, label=f"std (n_scenarios={n_scenarios})")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Turn t")
    ax.set_ylabel(f"ΔA_t (+: toward {value_a} | -: toward {value_b})")
    ax.legend(loc="best")
    ax.set_title(f"Model -> {value_model},  user -> {value_user}")


def plot_tradeoff_aggregate(alignment_repeat_stats_df, value_a, value_b, condition=""):
    if condition == "both":
        condition_keys = ["both_va", "both_vb"]
    elif condition == "opposite":
        condition_keys = ["model_va_user_vb", "model_vb_user_va"]
    elif condition == "only_user":
        condition_keys = ["user_va", "user_vb"]
    elif condition == "baseline":
        condition_keys = ["baseline"]
    else:
        raise ValueError("condition must be one of ['baseline', 'both', 'opposite', 'only_user']")

    df = alignment_repeat_stats_df.copy()
    df = df.dropna(subset=["A_t_mean", "turn"]).copy()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    df = df[df["condition"].isin(condition_keys)].copy()

    # restrict to exactly the one requested trade-off, regardless of a/b order in the data
    pair_mask = (
        ((df["scenario_value_a"] == value_a) & (df["scenario_value_b"] == value_b))
        | ((df["scenario_value_a"] == value_b) & (df["scenario_value_b"] == value_a))
    )
    df = df[pair_mask].copy()

    if df.empty:
        raise ValueError(f"No rows found for trade-off {value_a} vs {value_b} under condition='{condition}'")

    # normalize to however this pair is actually ordered in the data, so
    # downstream filtering against model_target_value/user_target_value matches
    actual_value_a = df["scenario_value_a"].iloc[0]
    actual_value_b = df["scenario_value_b"].iloc[0]

    trajectories = []
    if condition == "baseline":
        # baseline is a single, undirected condition, so only one panel is needed
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        trajectories.append(_plot_tradeoff_panel(ax, df, actual_value_a, actual_value_b, actual_value_a, condition))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, toward_value in zip(axes, [actual_value_a, actual_value_b]):
            trajectories.append(_plot_tradeoff_panel(ax, df, actual_value_a, actual_value_b, toward_value, condition))

    fig.suptitle(f"{actual_value_a} vs {actual_value_b} — aggregated across scenarios ({condition})", y=1.03)
    fig.tight_layout()
    plt.show()

    return trajectories


def _plot_tradeoff_panel(ax, subset, value_a, value_b, toward_value, condition):
    if condition == "both":
        value_model = toward_value
        value_user = toward_value
    elif condition == "opposite":
        value_model = toward_value
        values = [value_a, value_b]
        values.remove(toward_value)
        value_user = values[0]
    elif condition == "only_user":
        value_model = ""
        value_user = toward_value
    elif condition == "baseline":
        value_model = ""
        value_user = ""
    else:
        raise ValueError("condition must be one of ['baseline', 'both', 'opposite', 'only_user']")

    g_sel = subset[
        (subset["scenario_value_a"] == value_a)
        & (subset["scenario_value_b"] == value_b)
        & (subset["model_target_value"] == value_model)
        & (subset["user_target_value"] == value_user)
    ].copy()

    g_sel = g_sel.assign(
        combo_key=(g_sel["conversation_id"].astype(str) + "::" + g_sel["condition"].astype(str))
    )
    g_stats = (
        g_sel.groupby("turn", as_index=False)
        .agg(
            A_t_mean=("A_t_mean", "mean"),
            A_t_std=("A_t_mean", "std"),
            n_scenarios=("combo_key", "nunique"),
        )
        .sort_values("turn")
    )
    g_stats["A_t_std"] = g_stats["A_t_std"].fillna(0.0).clip(lower=0.0)

    x = g_stats["turn"].to_numpy()
    y = g_stats["A_t_mean"].to_numpy()
    s = g_stats["A_t_std"].to_numpy()
    n_scenarios = int(round(g_stats["n_scenarios"].mean()))

    ax.plot(x, y, marker="o", label=f"mean A_t (n={n_scenarios} scenarios)")
    ax.fill_between(x, y - s, y + s, alpha=0.2, label="± 1 std")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Turn t")
    ax.set_ylabel(f"Probe score A_t\n(− = toward {value_b}   |   + = toward {value_a})")

    if condition == "baseline":
        ax.set_title("Baseline")
    else:
        model_label = value_model if value_model else "not primed"
        ax.set_title(f"Assistant → {model_label}    |    User → {value_user}")
    ax.legend(loc="best")

    return {
        "value_assistant": toward_value,
        "A_t_mean": y,
        "A_t_std": s
    }

def _build_transcript_html(subset: pd.DataFrame, conv_id: str, repeat_idx: int, A_t_column: str) -> str:
    if subset.empty:
        return "<p>No turns found for this selection.</p>"

    conv_header = (
        f"<h3>Conversation: {escape(str(conv_id))}</h3>"
        f"<p><strong>Condition:</strong> {escape(str(subset['condition'].iloc[0]))}</p>"
        f"<p><strong>Run:</strong> {int(repeat_idx)}</p>"
    )
    blocks = [conv_header]
    for _, row in subset.sort_values("turn").iterrows():
        user_text = str(row.get("user_turn") or "")
        assistant_text = str(row.get("assistant_response") or "")
        blocks.append(
            "<div style='margin-bottom: 1.4em; padding: 0.8em; border: 1px solid #ddd; border-radius: 6px;'>"
            f"<p><strong>Turn {int(row['turn'])}</strong> | <strong>A_t:</strong> {float(row[A_t_column]):.4f}</p>"
            "<p><strong>User</strong></p>"
            f"<p><strong>Value_a:</strong> {row['scenario_value_a']} (logits={float(row['logit_a']):.4f}) | <strong>Value_b:</strong> {row['scenario_value_b']} (logits={float(row['logit_b']):.4f})</p>"
            "<p><strong>User</strong></p>"
            f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin: 0;'>{escape(user_text)}</pre>"
            "<p><strong>Assistant</strong></p>"
            f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin: 0;'>{escape(assistant_text)}</pre>"
            "</div>"
        )
    return "\n".join(blocks)


def _display_transcript(subset: pd.DataFrame, conv_id: str, repeat_idx: int, A_t_column: str) -> None:
    html = _build_transcript_html(subset, conv_id, repeat_idx, A_t_column)
    display(HTML(html), display_id="inspect_conversation_transcript")


def select_run_view_subset(run_view_df: pd.DataFrame, conv_id: str, repeat_idx: int, condition_name: str) -> pd.DataFrame:
    if run_view_df.empty:
        return run_view_df.iloc[0:0].copy()

    conv_id = str(conv_id).strip()
    repeat_idx = int(repeat_idx)
    condition_name = str(condition_name).strip()

    subset = run_view_df[
        (run_view_df["conversation_id"].astype(str).str.strip() == conv_id)
        & (run_view_df["repeat_index"].astype(int) == repeat_idx)
        & (run_view_df["condition"].astype(str).str.strip() == condition_name)
    ].copy()
    return subset.sort_values("turn").reset_index(drop=True)


def inspect_conversation(alignment_by_turn_df, run_records, use_delta_with_baseline=True):
    A_t_column = "delta_A_t" if use_delta_with_baseline else "A_t"

    text_lookup = {}
    for rec in run_records:
        conv_id = rec.get("conversation_id")
        condition_payload = rec.get("condition") or {}
        condition_name = None
        if isinstance(condition_payload, dict):
            condition_name = condition_payload.get("name") or condition_payload.get("label")
        elif condition_payload:
            condition_name = str(condition_payload)

        repeat_index = int(rec.get("repeat_index", 0))
        turns = rec.get("turns", [])
        responses = rec.get("responses", [])
        max_len = max(len(turns), len(responses), 0)
        for turn_idx in range(max_len):
            payload = {
                "user_turn": turns[turn_idx] if turn_idx < len(turns) else None,
                "assistant_response": responses[turn_idx] if turn_idx < len(responses) else None,
            }
            for condition_key in [condition_name, str(condition_payload) if condition_payload else None]:
                if condition_key:
                    text_lookup[(conv_id, condition_key, repeat_index, turn_idx)] = payload

    run_view_rows = []
    for _, row in alignment_by_turn_df.iterrows():
        conv_id = row["conversation_id"]
        condition = row["condition"]
        repeat_idx = int(row.get("repeat_index", 0))
        turn_idx = int(row["turn"])
        text_pair = text_lookup.get((conv_id, condition, repeat_idx, turn_idx), {})
        run_view_rows.append({
            "conversation_id": conv_id,
            "condition": condition,
            "repeat_index": repeat_idx,
            "turn": turn_idx,
            "scenario_value_a": row["scenario_value_a"],
            "scenario_value_b": row["scenario_value_b"],
            A_t_column: float(row[A_t_column]),
            "logit_a": float(row["logit_a"]),
            "logit_b": float(row["logit_b"]),
            "user_turn": text_pair.get("user_turn"),
            "assistant_response": text_pair.get("assistant_response"),
        })

    run_view_df = pd.DataFrame(run_view_rows).sort_values(["conversation_id", "condition", "repeat_index", "turn"]).reset_index(drop=True)

    conv_options = sorted(run_view_df["conversation_id"].astype(str).unique().tolist())
    run_options = sorted(run_view_df["repeat_index"].astype(int).unique().tolist())

    conv_dropdown = Dropdown(options=conv_options, description="Conversation")
    run_dropdown = Dropdown(options=run_options, description="Run")
    condition_dropdown = Dropdown(options=[], description="Condition")
    out = Output()
    suppress_callbacks = False

    def get_condition_options(conv_id: str, repeat_idx: int):
        subset = run_view_df[
            (run_view_df["conversation_id"].astype(str).str.strip() == str(conv_id).strip())
            & (run_view_df["repeat_index"].astype(int) == int(repeat_idx))
        ]
        return sorted(subset["condition"].astype(str).unique().tolist())

    def render_run(conv_id: str, repeat_idx: int, condition_name: str):
        subset = select_run_view_subset(run_view_df, conv_id, repeat_idx, condition_name)
        with out:
            out.clear_output(wait=True)
            _display_transcript(subset, conv_id, repeat_idx, A_t_column=A_t_column)

    def refresh_condition_options():
        nonlocal suppress_callbacks
        suppress_callbacks = True
        options = get_condition_options(conv_dropdown.value, int(run_dropdown.value))
        condition_dropdown.options = options
        if options:
            condition_dropdown.value = options[0]
        suppress_callbacks = False

    def on_conv_or_run_change(change):
        if suppress_callbacks:
            return
        if change["type"] == "change" and change["name"] == "value":
            refresh_condition_options()
            render_run(conv_dropdown.value, int(run_dropdown.value), condition_dropdown.value)

    def on_condition_change(change):
        if suppress_callbacks:
            return
        if change["type"] == "change" and change["name"] == "value":
            render_run(conv_dropdown.value, int(run_dropdown.value), condition_dropdown.value)

    conv_dropdown.observe(on_conv_or_run_change)
    run_dropdown.observe(on_conv_or_run_change)
    condition_dropdown.observe(on_condition_change)
    display(VBox([conv_dropdown, run_dropdown, condition_dropdown, out]), display_id="inspect_conversation_widget")

    refresh_condition_options()
    render_run(conv_dropdown.value, int(run_dropdown.value), condition_dropdown.value)
