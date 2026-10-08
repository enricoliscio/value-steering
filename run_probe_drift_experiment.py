#!/usr/bin/env python3
"""Run a controlled probe-drift experiment across multiple conversation conditions.

This script executes the same conversation set under different condition prompts,
records probe-axis drift trajectories, and writes aggregate comparisons.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from activation_drift import LlamaChatbot
from activation_drift.exp_setup import (
    load_conditions,
    load_conversations,
    set_all_seeds,
    write_csv,
    LocalSimulatorConfig,
    CloudSimulatorConfig,
    make_user_simulator,
    load_caa_vector,
    load_caa_vector_centroid,
    chat_with_oom_retry,
    compute_injection_drift_for_intervention,
    generate_injection_prompt,
    _find_tradeoff_entry,
    TARGET_MODEL_BASIC_INSTRUCTION
)
from activation_drift.utils import DEFAULT_LLM_MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled probe-drift experiment matrix")
    parser.add_argument(
        "--target-model-path",
        default=DEFAULT_LLM_MODEL_PATH,
        help="Path to the target assistant model being probed and analyzed",
    )

    parser.add_argument(
        "--target-model-quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional quantization for target assistant model when on CUDA",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained Schwartz probe checkpoint",
    )
    parser.add_argument(
        "--conversations",
        required=True,
        help="JSON file with conversation templates: [{id, turns:[...]}, ...]",
    )
    parser.add_argument(
        "--conditions",
        required=True,
        help="JSON file with conditions: [{name, system_prompt, user_instruction}, ...]",
    )
    parser.add_argument(
        "--assistant-steering-method",
        choices=["system_prompt", "caa", "injection", "softcsa"],
        default="system_prompt",
        help="How assistant steering is applied: system prompts (default), injection, or CAA activation addition or soft CSA addition?",
    )
    parser.add_argument(
        "--caa-vector-file",
        default=None,
        help="Path to saved CAA vector (.npy or layer-map .json). Required when --assistant-steering-method caa",
    )
    parser.add_argument(
        "--caa-alpha",
        type=float,
        default=1.0,
        help="Steering strength alpha for CAA (h'_l = h_l + alpha * a)",
        # for CSA, the scale alpha changes based on hidden vector alignment with the steering vector
    )
    parser.add_argument(
        "--injection-threshold",
        type=float,
        default=-0.5,
        help="Inject when the evaluator's score is below this threshold (e.g. -0.5 triggers on mild or strong drift; -1.5 triggers only on strong drift).",
    )
    parser.add_argument(
        "--injection-cooldown-turns",
        type=int,
        default=1,
        help="Number of turns after an injection during which injection is suppressed.",
    )
    parser.add_argument(
        "--injection-require-streak",
        type=int,
        default=1,
        help="Consecutive drifting evaluations required before injecting.",
    )
    parser.add_argument(
        "--injection-rubric",
        default="data/injection_evaluator_rubric.json",
        help="JSON file containing the rubric sent to the external injection evaluator.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-conversation-turns",
        type=int,
        default=15,
        help="Maximum number of assistant turns per conversation in simulated mode",
    )
    parser.add_argument(
        "--user-simulator",
        choices=["cloud", "local"],
        default="cloud",
        help="Choose whether to use local or cloud user model",
    )
    parser.add_argument(
        "--user-simulator-cloud-model",
        default="gemma4:31b-cloud",
        help="Cloud user model name",
    )
    parser.add_argument(
        "--user-simulator-cloud-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for cloud user simulator generation",
    )
    parser.add_argument(
        "--user-simulator-cloud-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for cloud user simulator",
    )
    parser.add_argument(
        "--user-simulator-cloud-top-p",
        type=float,
        default=0.9,
        help="Top-p for cloud user simulator",
    )
    parser.add_argument(
        "--user-simulator-cloud-debug-messages",
        action="store_true",
        help="Print compact role/message debug info for cloud user simulator",
    )
    parser.add_argument(
        "--use-centroid-activations",
        action="store_true",
        help="Use the centroid approach to compute activation differences?",
    )
    parser.add_argument(
        "--user-role-model-path",
        default="/home/USER/dev/models/meta-llama/Llama-3.2-3B-Instruct",
        help="Path to the model that simulates the user role",
    )
    parser.add_argument(
        "--user-simulator-device",
        default="cpu",
        help="Device for user simulator: cpu (default), auto, or cuda",
    )
    parser.add_argument("--user-simulator-max-new-tokens", type=int, default=4096)
    parser.add_argument("--user-simulator-temperature", type=float, default=0.8)
    parser.add_argument("--user-simulator-top-p", type=float, default=0.9)
    parser.add_argument(
        "--user-simulator-quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional quantization for simulator model when on CUDA",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument(
        "--num-seed-runs",
        type=int,
        default=5,
        help="Number of repeated runs per conversation-condition using fixed seeds",
    )
    parser.add_argument(
        "--assistant-oom-retries",
        type=int,
        default=2,
        help="Number of retries for assistant generation on CUDA OOM",
    )
    parser.add_argument(
        "--assistant-oom-backoff",
        type=float,
        default=0.5,
        help="Token backoff multiplier for each assistant OOM retry",
    )
    parser.add_argument(
        "--token-drift-mode",
        choices=["none", "prefix"],
        default="none",
        help="How to compute intra-turn drift traces for assistant responses",
    )
    parser.add_argument(
        "--token-drift-stride",
        type=int,
        default=4,
        help="Prefix step in tokens for token-drift tracing",
    )
    parser.add_argument(
        "--token-drift-max-points",
        type=int,
        default=64,
        help="Maximum sampled token-drift points per assistant turn (0 = no cap)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write run logs and summaries (default: artifacts/probe_experiment_TIMESTAMP)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    max_conversation_turns = int(args.max_conversation_turns)
    if max_conversation_turns < 1:
        raise ValueError("--max-conversation-turns must be >= 1")
    if int(args.num_seed_runs) < 1:
        raise ValueError("--num-seed-runs must be >= 1")
    if int(args.injection_cooldown_turns) < 0:
        raise ValueError("--injection-cooldown-turns must be >= 0")
    if int(args.injection_require_streak) < 1:
        raise ValueError("--injection-require-streak must be >= 1")
    if args.assistant_steering_method in ["caa", "softcsa"] and not args.caa_vector_file:
        raise ValueError("--caa-vector-file is required when --assistant-steering-method is caa or softcsa")

    seed_schedule = [int(args.seed + i) for i in range(int(args.num_seed_runs))]

    target_model_path = args.target_model_path or DEFAULT_LLM_MODEL_PATH
    user_role_model_path = args.user_role_model_path

    caa_layer_name = None
    caa_vector = None

    def _resolve_condition_caa_alpha(condition: Dict) -> tuple[float, str]:
        """Resolve signed per-condition CAA alpha from caa_direction only."""
        base = abs(float(args.caa_alpha))
        raw_direction = str(condition.get("caa_direction", "")).strip().lower()

        if not raw_direction:
            return 0.0, "missing"
        if raw_direction == "value_a":
            return base, "caa_direction"
        if raw_direction == "value_b":
            return -base, "caa_direction"

        raise ValueError(
            f"Invalid caa_direction '{raw_direction}'. Allowed values: value_a, value_b."
        )

    conversations = load_conversations(Path(args.conversations))

    prepared_runs = []
    conditions_path = Path(args.conditions) if args.conditions else None
    for conversation in conversations:
        scenario_value_a = str(conversation.get("value_a", "")).strip()
        scenario_value_b = str(conversation.get("value_b", "")).strip()
        if not scenario_value_a or not scenario_value_b:
            raise ValueError(
                f"Conversation '{conversation['id']}' must define both value_a and value_b"
            )
        if scenario_value_a == scenario_value_b:
            raise ValueError(f"Conversation '{conversation['id']}' has identical value_a and value_b")

        scenario_conditions = load_conditions(
            conditions_path,
            scenario_value_a,
            scenario_value_b,
        )
        prepared_runs.append(
            {
                "conversation": conversation,
                "scenario_value_a": scenario_value_a,
                "scenario_value_b": scenario_value_b,
                "conditions": scenario_conditions,
            }
        )
    caa_layer_name = None
    caa_vector = None
    if args.assistant_steering_method in ["caa", "softcsa"]:
        if args.use_centroid_activations:
            caa_layer_name, caa_vector = load_caa_vector_centroid(Path(args.caa_vector_file), value_a=prepared_runs[0]['scenario_value_a'], value_b=prepared_runs[0]['scenario_value_b'])
        else:  # if you're not using the centroid activations, then use the current pairwise strategy
            caa_layer_name, caa_vector = load_caa_vector(Path(args.caa_vector_file))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts") / f"probe_experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Probe Drift Experiment Matrix")
    print("=" * 80)
    print(f"Target model  : {target_model_path}")
    print(f"Target quant  : {args.target_model_quantization}")
    print(f"Checkpoint    : {args.checkpoint}")
    print("Probe axis    : scenario-specific from conversations JSON (value_a -> value_b)")
    print(f"Steering mode : {args.assistant_steering_method}")
    if args.assistant_steering_method == "caa":
        print(f"CAA vector    : {args.caa_vector_file}")
        print(f"CAA layer     : {caa_layer_name}")
        print(f"CAA alpha     : {args.caa_alpha}")
    print("User mode     : simulated")
    if args.user_simulator == "cloud":
        print(f"User model    : {args.user_simulator_cloud_model}")
    else:
        print(f"User model    : {user_role_model_path}")
    print(f"Max turns     : {max_conversation_turns}")
    print(f"User quant    : {args.user_simulator_quantization}")
    print(f"OOM retries   : {args.assistant_oom_retries} (backoff={args.assistant_oom_backoff})")
    print(
        f"Token drift   : {args.token_drift_mode} (stride={args.token_drift_stride}, max_points={args.token_drift_max_points})")
    print(f"Seed runs     : {args.num_seed_runs} ({seed_schedule[0]}..{seed_schedule[-1]})")
    print(f"Conversations : {len(conversations)}")
    cond_counts = sorted({len(item["conditions"]) for item in prepared_runs})
    print(f"Conditions/scn: {cond_counts if len(cond_counts) > 1 else cond_counts[0]}")
    base_configs = sum(len(item["conditions"]) for item in prepared_runs)
    print(f"Total configs : {base_configs}")
    print(f"Total runs    : {base_configs * int(args.num_seed_runs)}")
    print(f"Output dir    : {output_dir}")
    print("=" * 80)

    target_chatbot = LlamaChatbot(
        target_model_path,
        save_turn_artifacts=False,
        save_turn_activations=False,
        quantization=args.target_model_quantization,
    )

    if args.user_simulator == "local":
        config = LocalSimulatorConfig(
            model_path=user_role_model_path,
            device=args.user_simulator_device,
            quantization=args.user_simulator_quantization,
            max_new_tokens=args.user_simulator_max_new_tokens,
            temperature=args.user_simulator_temperature,
            top_p=args.user_simulator_top_p,
        )
        user_role_simulator = make_user_simulator(
            backend="local",
            local_config=config,
        )
    else:
        config = CloudSimulatorConfig(
            model_name=args.user_simulator_cloud_model,
            max_new_tokens=args.user_simulator_cloud_max_new_tokens,
            temperature=args.user_simulator_cloud_temperature,
            top_p=args.user_simulator_cloud_top_p,
            debug_messages=args.user_simulator_cloud_debug_messages,
        )
        user_role_simulator = make_user_simulator(
            backend="cloud",
            cloud_config=config,
        )

    active_probe_axis = (None, None)

    run_rows: List[Dict] = []
    probe_rows: List[Dict] = []
    run_jsonl_path = output_dir / "runs.jsonl"

    def _write_progress_artifacts(completed_conversations: int, is_final: bool = False) -> str | None:
        """Write recoverable progress artifacts from currently accumulated rows."""
        write_csv(output_dir / "probe_scores_by_turn.csv", probe_rows)

        config_dump = {
            "target_model_path": target_model_path,
            "target_model_quantization": args.target_model_quantization,
            "model_path": target_model_path,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "num_seed_runs": int(args.num_seed_runs),
            "seed_schedule": seed_schedule,
            "max_conversation_turns": max_conversation_turns,
            "user_role_model_path": user_role_model_path,
            "user_mode": "simulated",
            "user_simulator_device": args.user_simulator_device,
            "user_simulator_cloud_model": args.user_simulator_cloud_model,
            "user_simulator_cloud_max_new_tokens": args.user_simulator_cloud_max_new_tokens,
            "user_simulator_cloud_temperature": args.user_simulator_cloud_temperature,
            "user_simulator_cloud_top_p": args.user_simulator_cloud_top_p,
            "user_simulator_cloud_debug_messages": bool(args.user_simulator_cloud_debug_messages),
            "user_simulator_max_new_tokens": args.user_simulator_max_new_tokens,
            "user_simulator_temperature": args.user_simulator_temperature,
            "user_simulator_top_p": args.user_simulator_top_p,
            "user_simulator_quantization": args.user_simulator_quantization,
            "assistant_oom_retries": args.assistant_oom_retries,
            "assistant_oom_backoff": args.assistant_oom_backoff,
            "token_drift_mode": args.token_drift_mode,
            "token_drift_stride": args.token_drift_stride,
            "token_drift_max_points": args.token_drift_max_points,
            "max_new_tokens": args.max_new_tokens,
            "conversations_file": args.conversations,
            "conditions_file": args.conditions,
            "steering_mode": args.assistant_steering_method,
            "caa_vector_file": args.caa_vector_file,
            "caa_alpha": args.caa_alpha,
            "caa_layer_name": caa_layer_name,
            "conditions_by_conversation": {
                item["conversation"]["id"]: item["conditions"]
                for item in prepared_runs
            },
            "num_conversations": len(conversations),
            "completed_conversations": int(completed_conversations),
            "total_configs": base_configs,
            "total_runs": len(run_rows),
            "status": "complete" if is_final else "in_progress",
        }

        partial_path = output_dir / "experiment_config.partial.json"
        partial_path.write_text(
            json.dumps(config_dump, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        if is_final:
            (output_dir / "experiment_config.json").write_text(
                json.dumps(config_dump, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

    injection_rubric_config = None
    if args.assistant_steering_method == "injection":
        rubric_path = Path(args.injection_rubric)
        if not rubric_path.is_file():
            raise ValueError(f"Injection rubric file not found: {rubric_path}")
        injection_rubric_text = rubric_path.read_text(encoding="utf-8")
        try:
            injection_rubric_config = json.loads(injection_rubric_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Injection rubric must be valid JSON: {rubric_path}") from exc
        if not isinstance(injection_rubric_config, dict) or not injection_rubric_config.get("tradeoffs"):
            raise ValueError(
                "Injection rubric must be a JSON object with a non-empty 'tradeoffs' field"
            )
        score_scale = injection_rubric_config.get("score_scale")
        if not isinstance(score_scale, dict) or not score_scale:
            raise ValueError("Injection rubric must define a non-empty 'score_scale' object")
        try:
            score_levels = {str(level): int(level) for level in score_scale}
        except (TypeError, ValueError) as exc:
            raise ValueError("Injection rubric score_scale keys must be integers") from exc
        numeric_score_levels = set(score_levels.values())
        if not min(numeric_score_levels) <= args.injection_threshold <= max(numeric_score_levels):
            print(
                f"Warning: injection threshold {args.injection_threshold} is outside the rubric score range "
                f"[{min(numeric_score_levels)}, {max(numeric_score_levels)}]."
            )
        for prepared in prepared_runs:
            for condition in prepared["conditions"]:
                target_value = condition.get("model_target_value", "")
                if not target_value:
                    continue
                tradeoff_key, tradeoff_entry = _find_tradeoff_entry(
                    injection_rubric_config,
                    prepared["scenario_value_a"],
                    prepared["scenario_value_b"],
                )
                if not isinstance(tradeoff_entry, dict):
                    raise ValueError(
                        f"Conversation '{prepared['conversation']['id']}', condition '{condition['name']}' "
                        f"has a malformed rubric trade-off entry '{tradeoff_key}'"
                    )
                values_block = tradeoff_entry.get("values", {})
                target_criteria = values_block.get(target_value) if isinstance(values_block, dict) else None
                if not isinstance(target_criteria, dict) or not target_criteria:
                    raise ValueError(
                        f"Conversation '{prepared['conversation']['id']}', condition '{condition['name']}' "
                        f"has no non-empty rubric criteria for target value '{target_value}' "
                        f"in trade-off '{tradeoff_key}'"
                    )
                invalid_levels = set(target_criteria) - set(score_levels)
                if invalid_levels:
                    raise ValueError(
                        f"Conversation '{prepared['conversation']['id']}', condition '{condition['name']}' "
                        f"has rubric levels not present in score_scale: {sorted(invalid_levels)}"
                    )
                missing_levels = set(score_levels) - set(target_criteria)
                if missing_levels:
                    raise ValueError(
                        f"Conversation '{prepared['conversation']['id']}', condition '{condition['name']}' "
                        f"is missing rubric levels: {sorted(missing_levels, key=int, reverse=True)}"
                    )
                empty_levels = [
                    level for level, description in target_criteria.items()
                    if not isinstance(description, str) or not description.strip()
                ]
                if empty_levels:
                    raise ValueError(
                        f"Conversation '{prepared['conversation']['id']}', condition '{condition['name']}' "
                        f"has empty rubric criteria at levels: {sorted(empty_levels)}"
                    )

    with run_jsonl_path.open("w", encoding="utf-8") as fjsonl:
        for conv_idx, conversation in enumerate(conversations):
            prepared = prepared_runs[conv_idx]
            conversation = prepared["conversation"]
            scenario_conditions = prepared["conditions"]

            conv_id = conversation["id"]
            turns = conversation["turns"]
            scenario = conversation.get("scenario", "")
            scenario_tradeoff_instructions = conversation["tradeoff_instructions"]
            scenario_key = conversation.get("scenario_key", conv_id)
            scenario_value_a = prepared["scenario_value_a"]
            scenario_value_b = prepared["scenario_value_b"]
            if active_probe_axis != (scenario_value_a, scenario_value_b):
                target_chatbot.set_schwartz_probe_axis(
                    scenario_value_a,
                    scenario_value_b,
                    checkpoint_path=args.checkpoint,
                )
                active_probe_axis = (scenario_value_a, scenario_value_b)

            print(f"\nConversation {conv_idx + 1}/{len(conversations)}: {conv_id}")
            print(f"  trade-off: {scenario_value_a} vs {scenario_value_b}")

            for cond_idx, condition in enumerate(scenario_conditions):
                for repeat_index, run_seed in enumerate(seed_schedule):
                    set_all_seeds(run_seed)

                    print(f"Generating conversation under condition: {condition['name']} (seed {run_seed})")

                    model_instruction = str(condition.get("model_instruction", ""))
                    caa_alpha_effective = None
                    caa_direction_source = None
                    if args.assistant_steering_method == "caa":
                        caa_alpha_effective, caa_direction_source = _resolve_condition_caa_alpha(condition=condition)
                        # Keep assistant baseline prompt clean when steering via activations.
                        target_chatbot.set_activation_steering(
                            layer_name=caa_layer_name,
                            steering_vector=caa_vector,
                            alpha=float(caa_alpha_effective),
                        )
                        model_instruction = ""
                    elif args.assistant_steering_method == "softcsa":
                        caa_alpha_effective, caa_direction_source = _resolve_condition_caa_alpha(condition=condition)
                        # soft csa assumes your steering vector is always toward the value you want to steer towards.
                        # So I reverted the sign change from the resolve_condition since the directional correction happens inside automatically.
                        sign = 1.0 if caa_alpha_effective > 0 else -1.0
                        target_chatbot.set_activation_steering_softcsa(
                            layer_name=caa_layer_name,
                            steering_vector=caa_vector * sign,
                            alpha=float(abs(caa_alpha_effective)))
                        model_instruction = ""
                    else:
                        target_chatbot.clear_activation_steering()

                    target_chatbot.conversation_history = []
                    target_chatbot.drift_scores_history = []

                    system_prompt = TARGET_MODEL_BASIC_INSTRUCTION
                    if model_instruction.strip():
                        system_prompt = "\n".join((TARGET_MODEL_BASIC_INSTRUCTION, model_instruction.strip()))

                    target_chatbot.conversation_history.append({"role": "system", "content": system_prompt})

                    responses = []
                    user_turns = []
                    token_drift_traces = []
                    alignment_trajectory: List[float] = []
                    probe_scores_trajectory: List[Dict] = []
                    injection_evaluations: List[Dict] = []
                    last_injection_turn = None
                    consecutive_drift_count = 0

                    target_turns = max_conversation_turns
                    seed_turns = list(turns)
                    for turn_idx in tqdm(range(target_turns)):
                        if turn_idx < len(seed_turns):
                            user_turn = seed_turns[turn_idx]
                        else:
                            user_turn = user_role_simulator.generate_user_turn(
                                scenario=scenario,
                                conversation_history=target_chatbot.conversation_history,
                                user_instruction=condition["user_instruction"],
                                scenario_tradeoff_instruction=scenario_tradeoff_instructions
                            )

                        user_turns.append(user_turn)

                        responses.append(
                            chat_with_oom_retry(
                                chatbot=target_chatbot,
                                user_turn=user_turn,
                                max_new_tokens=args.max_new_tokens,
                                oom_retries=args.assistant_oom_retries,
                                oom_backoff=args.assistant_oom_backoff,
                            )
                        )
                        response, used_max_new_tokens = responses[-1]
                        responses[-1] = response
                        if args.token_drift_mode == "prefix":
                            trace = target_chatbot.compute_token_drift_for_last_response(
                                stride=args.token_drift_stride,
                                max_points=None if args.token_drift_max_points <= 0 else args.token_drift_max_points,
                            )
                        else:
                            trace = []
                        token_drift_traces.append(
                            {
                                "turn": len(user_turns) - 1,
                                "assistant_max_new_tokens_used": int(used_max_new_tokens),
                                "trace": trace,
                            }
                        )

                        analyzer = target_chatbot.drift_analyzer
                        if analyzer is None:
                            raise RuntimeError("Schwartz probe analyzer is not configured")

                        logits_by_value = analyzer.predict_logits(target_chatbot.probe.activations)
                        probs_by_value = analyzer.predict_scores(target_chatbot.probe.activations)

                        value_a_key = analyzer.value_a
                        value_b_key = analyzer.value_b
                        alignment_value = float(logits_by_value[value_a_key] - logits_by_value[value_b_key])
                        alignment_trajectory.append(alignment_value)

                        turn_id = len(user_turns) - 1
                        probe_scores_trajectory.append(
                            {
                                "turn": int(turn_id),
                                "value_a": value_a_key,
                                "value_b": value_b_key,
                                "logits": logits_by_value,
                                "probs": probs_by_value,
                            }
                        )

                        for value_name in analyzer.schwartz_values:
                            probe_rows.append(
                                {
                                    "conversation_id": conv_id,
                                    "scenario_key": scenario_key,
                                    "scenario_value_a": scenario_value_a,
                                    "scenario_value_b": scenario_value_b,
                                    "steering_mode": args.assistant_steering_method,
                                    "condition": condition["name"],
                                    "model_target_value": condition.get("model_target_value", ""),
                                    "user_target_value": condition.get("user_target_value", ""),
                                    "user_mode": "simulated",
                                    "repeat_index": int(repeat_index),
                                    "seed": run_seed,
                                    "turn": int(turn_id),
                                    "value_name": value_name,
                                    "probe_logit": float(logits_by_value[value_name]),
                                    "probe_prob": float(probs_by_value[value_name]),
                                }
                            )

                        injection_target_value = condition.get("model_target_value", "")
                        if args.assistant_steering_method == "injection" and injection_target_value:
                            evaluation = compute_injection_drift_for_intervention(
                                conversation_history=target_chatbot.conversation_history,
                                scenario=scenario,
                                scenario_tradeoff_instruction=scenario_tradeoff_instructions,
                                value_a=scenario_value_a,
                                value_b=scenario_value_b,
                                target_value=injection_target_value,
                                rubric_config=injection_rubric_config,
                            )
                            is_drifting = evaluation["value"] < args.injection_threshold
                            skipped_reason = None
                            injected = False
                            drift_count_at_evaluation = consecutive_drift_count
                            if (
                                last_injection_turn is not None
                                and turn_id - last_injection_turn <= args.injection_cooldown_turns
                            ):
                                skipped_reason = "cooldown"
                            else:
                                consecutive_drift_count = (
                                    consecutive_drift_count + 1 if is_drifting else 0
                                )
                                drift_count_at_evaluation = consecutive_drift_count
                                if consecutive_drift_count >= args.injection_require_streak:
                                    injected = True
                                    last_injection_turn = turn_id
                                    consecutive_drift_count = 0
                                elif is_drifting:
                                    skipped_reason = "streak"

                            injection_evaluations.append(
                                {
                                    "turn": int(turn_id),
                                    "value": evaluation["value"],
                                    "reasoning": evaluation["reasoning"],
                                    "target_value": injection_target_value,
                                    "injected": injected,
                                    "skipped_reason": skipped_reason,
                                    "consecutive_drift_count": int(drift_count_at_evaluation),
                                }
                            )
                            if injected:
                                target_chatbot.conversation_history.append(
                                    {
                                        "role": "system",
                                        "content": generate_injection_prompt(
                                            value_a=scenario_value_a,
                                            value_b=scenario_value_b,
                                            target_value=injection_target_value,
                                            drift_value=evaluation["value"],
                                            judge_reasoning=evaluation["reasoning"],
                                        ),
                                    }
                                )
                    row = {
                        "conversation_id": conv_id,
                        "scenario_key": scenario_key,
                        "scenario_value_a": scenario_value_a,
                        "scenario_value_b": scenario_value_b,
                        "steering_mode": args.assistant_steering_method,
                        "condition": condition["name"],
                        "model_target_value": condition.get("model_target_value", ""),
                        "user_target_value": condition.get("user_target_value", ""),
                        "user_mode": "simulated",
                        "repeat_index": int(repeat_index),
                        "seed": run_seed,
                    }
                    run_rows.append(row)

                    run_detail = {
                        "conversation_id": conv_id,
                        "scenario_key": scenario_key,
                        "scenario_value_a": scenario_value_a,
                        "scenario_value_b": scenario_value_b,
                        "steering_mode": args.assistant_steering_method,
                        "condition": condition,
                        "assistant_system_prompt_applied": model_instruction,
                        "caa_alpha_effective": caa_alpha_effective,
                        "caa_direction_source": caa_direction_source,
                        "caa_direction_value": condition.get("caa_direction", ""),
                        "model_target_value": condition.get("model_target_value", ""),
                        "user_target_value": condition.get("user_target_value", ""),
                        "user_mode": "simulated",
                        "repeat_index": int(repeat_index),
                        "seed": run_seed,
                        "scenario": scenario,
                        "scenario_tradeoff_instructions": scenario_tradeoff_instructions,
                        "turns": user_turns,
                        "responses": responses,
                        "value_a": target_chatbot.drift_analyzer.value_a,
                        "value_b": target_chatbot.drift_analyzer.value_b,
                        "alignment_trajectory": alignment_trajectory,
                        "probe_scores_by_turn": probe_scores_trajectory,
                        "injection_evaluations": injection_evaluations,
                        "token_drift_traces": token_drift_traces,
                    }
                    fjsonl.write(json.dumps(run_detail, ensure_ascii=True) + "\n")

                # Persist progress after each completed condition to reduce loss on long runs.
                fjsonl.flush()
                _write_progress_artifacts(
                    completed_conversations=conv_idx,
                    is_final=False,
                )
                print(f"progress saved: {cond_idx + 1}/{len(scenario_conditions)} conditions.")

            # Persist progress after each completed scenario to reduce loss on long runs.
            fjsonl.flush()
            _write_progress_artifacts(
                completed_conversations=conv_idx + 1,
                is_final=False,
            )
            print(
                f"  progress saved: {conv_idx + 1}/{len(conversations)} scenarios "
                f"({len(run_rows)} runs)."
            )

    _write_progress_artifacts(
        completed_conversations=len(conversations),
        is_final=True,
    )

    print("\n" + "=" * 80)
    print("Experiment complete")
    print("=" * 80)
    print(f"Probe scores by turn: {output_dir / 'probe_scores_by_turn.csv'}")
    print(f"Full run details    : {run_jsonl_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
