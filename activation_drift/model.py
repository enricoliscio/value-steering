import numpy as np
from typing import Dict, List, Optional
import torch

from .utils import *
from .probe import ActivationProbe
from .activations import SteeringVectorAnalyzer
from .schwartz_probe import SchwartzProbeAxisAnalyzer


class LlamaChatbot:
    """Interactive Llama chatbot with activation extraction."""

    def __init__(
            self,
            model_path: str,
            save_turn_artifacts: bool = True,
            save_turn_activations: bool = True,
            quantization: str = "none",
            device: str = ""
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_path: Path to the Llama model directory
        """
        self.device = device if device else get_device()
        print(f"Loading model from {model_path}...")

        # Load tokenizer and model
        self.tokenizer = load_tokenizer(model_path)
        self.model = load_model(model_path, self.device, quantization=quantization)

        # Initialize activation probe
        self.probe = ActivationProbe(self.model)

        self.save_turn_artifacts = bool(save_turn_artifacts)
        self.save_turn_activations = bool(save_turn_activations)

        # Create output directory for activations
        self.activation_dir = get_path("activations")
        if self.save_turn_artifacts:
            make_directory(self.activation_dir)

        # Initialize conversation history
        self.conversation_history = []
        self.interaction_count = 0

        # Initialize drift tracking (optional)
        self.drift_analyzer = None
        self.drift_mode = None
        self.drift_scores_history = []  # Track drift scores over conversation

        # Optional activation-addition intervention (CAA-style).
        self._steering_hook_handle = None
        self._steering_layer_name = None
        self._steering_vector = None
        self._steering_alpha = 0.0

        print(f"✓ Model loaded on {self.device}")
        print("✓ Activation probe initialized")

    def _get_submodule_by_name(self, module_name: str):
        try:
            return self.model.get_submodule(module_name)
        except Exception as e:
            raise ValueError(f"Could not resolve submodule '{module_name}' on model: {e}")

    def clear_activation_steering(self):
        """Disable any active activation-addition intervention."""
        if self._steering_hook_handle is not None:
            self._steering_hook_handle.remove()
            self._steering_hook_handle = None
        self._steering_layer_name = None
        self._steering_vector = None
        self._steering_alpha = 0.0

    def set_activation_steering(self, layer_name: str, steering_vector, alpha: float = 1.0):
        """Enable CAA-style activation steering on a specific layer.

        Args:
            layer_name: Fully-qualified module name, e.g. model.layers.24
            steering_vector: 1D vector with hidden_size elements
            alpha: Steering strength multiplier
        """
        self.clear_activation_steering()

        vector_np = np.asarray(steering_vector, dtype=np.float32)
        if vector_np.ndim != 1:
            raise ValueError(f"steering_vector must be 1D, got shape {vector_np.shape}")

        layer_module = self._get_submodule_by_name(layer_name)
        self._steering_layer_name = layer_name
        self._steering_alpha = float(alpha)

        def _hook(_module, _inputs, output):
            # Decoder layers usually return tuples where output[0] is hidden states.
            if isinstance(output, tuple):
                if not output:
                    return output
                hidden = output[0]
                if hidden is None:
                    return output
                vec = self._steering_vector.to(device=hidden.device, dtype=hidden.dtype)
                steered = hidden + self._steering_alpha * vec.view(1, 1, -1)
                return (steered, *output[1:])

            # Fallback for modules that return hidden states directly.
            if torch.is_tensor(output):
                vec = self._steering_vector.to(device=output.device, dtype=output.dtype)
                return output + self._steering_alpha * vec.view(1, 1, -1)

            return output

        self._steering_vector = torch.tensor(vector_np, dtype=torch.float32, device=self.model.device)
        self._steering_hook_handle = layer_module.register_forward_hook(_hook)
        print(
            f"✓ Activation steering enabled on {layer_name} "
            f"(alpha={self._steering_alpha}, dim={vector_np.shape[0]})"
        )

    def set_activation_steering_softcsa(self, layer_name: str, steering_vector, alpha: float = 1.0):
        """Enable Soft CSA-style activation steering on a specific layer.

        Args:
            layer_name: Fully-qualified module name, e.g. model.layers.24
            steering_vector: 1D vector with hidden_size elements
            alpha: Base Steering strength multiplier
        """
        self.clear_activation_steering()

        vector_np = np.asarray(steering_vector, dtype=np.float32)
        if vector_np.ndim != 1:
            raise ValueError(f"steering_vector must be 1D, got shape {vector_np.shape}")

        layer_module = self._get_submodule_by_name(layer_name)
        self._steering_layer_name = layer_name
        self._steering_alpha = float(alpha)

        def _hook(_module, _inputs, output):
            # Decoder layers usually return tuples where output[0] is hidden states.
            if isinstance(output, tuple):
                if not output:
                    return output
                hidden = output[0]
                if hidden is None:
                    return output

                # since we fixed the sign convention at the calling function level,
                # the steering vector is always going to point in the direction of the value you want to steer towards.
                vec = self._steering_vector.to(device=hidden.device, dtype=hidden.dtype)
                align = torch.einsum('bsh,h->bs', hidden, vec)
                gate = torch.sigmoid(-align)
                coeff = self._steering_alpha * gate  # this rescales alpha to alpha_prime
                steered = hidden + coeff.unsqueeze(-1) * vec.view(1, 1, -1)
                return (steered, *output[1:])

            # Fallback for modules that return hidden states directly.
            if torch.is_tensor(output):
                vec = self._steering_vector.to(device=output.device, dtype=output.dtype)
                hidden = output
                align = torch.einsum('bsh,h->bs', hidden, vec)
                gate = torch.sigmoid(-align)
                coeff = self._steering_alpha * gate  # this rescales alpha to alpha_prime
                return output + coeff.unsqueeze(-1) * vec.view(1, 1, -1)

            return output

        self._steering_vector = torch.tensor(vector_np, dtype=torch.float32, device=self.model.device)
        self._steering_hook_handle = layer_module.register_forward_hook(_hook)
        print(
            f"✓ Activation steering enabled on {layer_name} "
            f"(alpha={self._steering_alpha}, dim={vector_np.shape[0]})"
        )

    def _reset_drift_tracking(self, analyzer, mode: str, behavior1: str, behavior2: str):
        """Set active drift analyzer and clear prior drift history."""
        self.behavior1 = behavior1
        self.behavior2 = behavior2
        self.drift_analyzer = analyzer
        self.drift_mode = mode
        self.drift_scores_history = []

    def chat(self, user_input: str, max_new_tokens: int = 128) -> str:
        """
        Process user input and generate a response with activation extraction.
        
        Args:
            user_input: User's message
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Model's response
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format conversation as prompt
        prompt = self._format_prompt()

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Clear previous activations
        self.probe.clear()

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        # Only decode the generated tokens (not the prompt)
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Add assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        # Recompute activations on the completed turn with a plain forward pass.
        # This keeps drift scoring aligned with how probe-training activations were extracted.
        self._refresh_turn_activations()

        # Save drift and optional interaction artifacts.
        self._save_interaction_data(user_input, response)

        self.interaction_count += 1

        return response

    def _custom_chat_template(self):
        prompt = ""

        for message in self.conversation_history:
            prompt += f"{message["role"]}:\n{message["content"]}\n\n"
        return prompt.removesuffix("\n\n")

    def _format_prompt(self, add_generation_prompt: bool = True):
        if add_generation_prompt:
            return self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return self._custom_chat_template()

    def _refresh_turn_activations(self):
        """Capture activations for the completed conversation turn with a standard forward pass."""
        scoring_prompt = self._format_prompt(add_generation_prompt=False)
        scoring_inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)

        self.probe.clear()
        with torch.no_grad():
            self.model(**scoring_inputs, use_cache=False)

    def set_steering_vector(self, behavior1: str, behavior2: str,
                            activation_base_dir: str = "behavior_activations",
                            pooling: int = 1):
        """
        Set up steering vector analysis for measuring activation drift.
        
        Args:
            behavior1: First behavior type (direction will be behavior1 -> behavior2)
            behavior2: Second behavior type
            activation_base_dir: Base directory containing saved activations
            pooling: -1 full-sequence mean, 1 last token, k mean over last k tokens
        """
        print(f"Setting up steering vector: {behavior1} -> {behavior2} (pooling={pooling})")
        analyzer = SteeringVectorAnalyzer(activation_base_dir)
        analyzer.compute_steering_vector(behavior1, behavior2, pooling=pooling)
        self._reset_drift_tracking(analyzer, mode="steering", behavior1=behavior1, behavior2=behavior2)
        print("✓ Ready to track activation drift")

    def set_valuenet_positive_steering(self, value_a: str, value_b: str,
                                       activation_base_dir: str = "activations/valuenet",
                                       pooling: int = 1):
        """Set steering vector between two ValueNet values using only positive examples.

        Args:
            value_a: First ValueNet value (e.g., ACHIEVEMENT)
            value_b: Second ValueNet value (e.g., BENEVOLENCE)
            activation_base_dir: Directory containing VALUE_positive folders
            pooling: -1 full-sequence mean, 1 last token, k mean over last k tokens
        """
        base_dir = get_path(activation_base_dir)
        if not base_dir.exists():
            raise ValueError(f"ValueNet activation directory not found: {base_dir}")

        positive_folders = [p.name for p in base_dir.iterdir() if p.is_dir() and p.name.endswith("_positive")]
        if not positive_folders:
            raise ValueError(f"No *_positive folders found in {base_dir}")

        # Build flexible lookup for value names (supports hyphen/underscore variations).
        value_to_folder = {}
        for folder in positive_folders:
            value_name = folder[:-len("_positive")]
            variants = {
                value_name,
                value_name.upper(),
                value_name.replace("-", "_"),
                value_name.replace("-", "_").upper(),
                value_name.replace("_", "-"),
                value_name.replace("_", "-").upper(),
            }
            for variant in variants:
                value_to_folder[variant] = folder

        def resolve_folder(value: str) -> str:
            raw = value.strip()
            if raw.endswith("_positive") and raw in positive_folders:
                return raw

            candidates = {
                raw,
                raw.upper(),
                raw.replace("-", "_"),
                raw.replace("-", "_").upper(),
                raw.replace("_", "-"),
                raw.replace("_", "-").upper(),
            }
            for candidate in candidates:
                if candidate in value_to_folder:
                    return value_to_folder[candidate]

            available_values = sorted(folder[:-len("_positive")] for folder in positive_folders)
            raise ValueError(
                f"Value '{value}' not found. Available values: {', '.join(available_values)}"
            )

        behavior1 = resolve_folder(value_a)
        behavior2 = resolve_folder(value_b)

        print(f"Using ValueNet positive folders: {behavior1} vs {behavior2}")
        self.set_steering_vector(
            behavior1=behavior1,
            behavior2=behavior2,
            activation_base_dir=activation_base_dir,
            pooling=pooling,
        )

    def set_schwartz_probe_axis(self, value_a: str, value_b: str,
                                checkpoint_path: str = "artifacts/schwartz_probe.pt"):
        """Use a trained Schwartz probe to measure drift along a value axis."""
        print(f"Loading Schwartz probe axis: {value_a} -> {value_b}")
        analyzer = SchwartzProbeAxisAnalyzer(checkpoint_path, value_a, value_b)
        self._reset_drift_tracking(
            analyzer,
            mode="probe",
            behavior1=analyzer.value_a,
            behavior2=analyzer.value_b,
        )
        print(
            f"✓ Ready to track probe drift on {analyzer.layer_name} "
            f"(pooling={analyzer.pooling}, checkpoint={checkpoint_path})"
        )

    def _save_interaction_data(self, user_input: str, response: str):
        """Save drift and optional artifacts for the current interaction."""
        timestamp = get_timestamp()

        # Compute drift scores if steering vectors are available.
        drift_scores = None
        if self.drift_analyzer:
            drift_scores = self.drift_analyzer.measure_activation_drift(self.probe.activations)
            self.drift_scores_history.append({
                "turn": self.interaction_count,
                "timestamp": timestamp,
                "user_input": user_input,
                "drift_scores": drift_scores,
            })

        # Optionally skip disk writes entirely (used by experiment runner).
        if not self.save_turn_artifacts:
            return

        interaction_id = f"interaction_{self.interaction_count}_{timestamp}"
        interaction_dir = self.activation_dir / interaction_id
        make_directory(interaction_dir)

        if self.save_turn_activations:
            activation_file = interaction_dir / "activations.pkl"
            self.probe.save_activations(str(activation_file))

        # Save metadata
        metadata = {
            "interaction_number": self.interaction_count,
            "timestamp": timestamp,
            "user_input": user_input,
            "model_response": response,
            "num_layers": len(self.probe.activations),
            "layer_names": list(self.probe.activations.keys()),
            "drift_scores": drift_scores,  # Include drift scores if available
            "activations_saved": bool(self.save_turn_activations),
        }
        metadata_file = interaction_dir / "metadata.json"
        save_json(metadata, metadata_file)

        print(f"✓ Interaction metadata saved to {interaction_dir}")

        # Print drift scores if available
        if drift_scores:
            self._print_drift_scores(drift_scores)

    def compute_token_drift_for_last_response(
            self,
            stride: int = 1,
            max_points: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Compute token-prefix drift trace for the latest assistant turn.

        This replays assistant response prefixes and measures drift on each prefix.
        """
        if not self.drift_analyzer:
            return []
        if len(self.conversation_history) < 2:
            return []
        if self.conversation_history[-1]["role"] != "assistant":
            return []

        response_text = self.conversation_history[-1]["content"]
        token_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        if not token_ids:
            return []

        step = max(1, int(stride))
        selected = list(range(step, len(token_ids) + 1, step))
        if not selected or selected[-1] != len(token_ids):
            selected.append(len(token_ids))

        if max_points is not None and int(max_points) > 0 and len(selected) > int(max_points):
            idx = np.linspace(0, len(selected) - 1, int(max_points)).round().astype(int)
            selected = [selected[i] for i in sorted(set(idx.tolist()))]
            if not selected or selected[-1] != len(token_ids):
                selected.append(len(token_ids))

        trace: List[Dict[str, float]] = []
        base_history = list(self.conversation_history)

        for token_count in selected:
            prefix_ids = token_ids[:token_count]
            prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
            temp_history = list(base_history)
            temp_history[-1] = {"role": "assistant", "content": prefix_text}

            scoring_prompt = self.tokenizer.apply_chat_template(
                temp_history,
                tokenize=False,
                add_generation_prompt=False,
            )
            scoring_inputs = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)

            self.probe.clear()
            with torch.no_grad():
                self.model(**scoring_inputs, use_cache=False)

            drift_scores = self.drift_analyzer.measure_activation_drift(self.probe.activations)
            drift_value = float(np.mean(list(drift_scores.values())))
            trace.append({
                "token_index": int(token_count),
                "drift": drift_value,
            })

        # Restore full-turn activations for consistency after prefix replay.
        self._refresh_turn_activations()
        return trace

    def run_interactive(self):
        """Run the interactive chat loop."""
        print("\n" + "=" * 60)
        print("Llama Chatbot with Activation Extraction")
        print("=" * 60)
        print("Commands:")
        print("  'quit' or 'exit'     - End conversation")
        print("  'history'            - Show conversation history")
        print("  'drift'              - Show activation drift evolution")
        print("  'steer BEHAVIOR1 BEHAVIOR2 [POOLING]' - Load steering vectors")
        print("                           pooling: default=1 (last token)")
        print("                           -1=full mean, 1=last token, k=last k mean")
        print("  'steer_values VALUE_A VALUE_B [POOLING]' - ValueNet positive-only steering")
        print("  'probe_values VALUE_A VALUE_B [CHECKPOINT]' - Schwartz probe axis drift")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if user_input.lower() == 'history':
                    self._print_history()
                    continue

                if user_input.lower().startswith('drift'):
                    parts = user_input.split()
                    if len(parts) == 1:
                        self.print_drift_evolution()
                    elif len(parts) == 2:
                        self.print_drift_evolution(parts[1])
                    else:
                        print("Usage: drift [LAYER]")
                        print("  Examples: drift")
                        print("            drift 20")
                        print("            drift model.layers.20")
                    continue

                if user_input.lower().startswith('steer_values '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        value_a = parts[1]
                        value_b = parts[2]
                        try:
                            pooling = int(parts[3]) if len(parts) >= 4 else 1
                        except ValueError:
                            print("Usage: steer_values VALUE_A VALUE_B [POOLING]")
                            print("  POOLING must be -1, 1, or any positive integer k")
                            continue

                        if len(parts) < 4:
                            print("Using default pooling=1 (last token)")

                        self.set_valuenet_positive_steering(value_a, value_b, pooling=pooling)
                    else:
                        print("Usage: steer_values VALUE_A VALUE_B [POOLING]")
                        print("  Example: steer_values ACHIEVEMENT BENEVOLENCE")
                        print("           steer_values SELF-DIRECTION POWER 8")
                    continue

                if user_input.lower().startswith('probe_values '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        value_a = parts[1]
                        value_b = parts[2]
                        checkpoint_path = parts[3] if len(parts) >= 4 else "artifacts/schwartz_probe.pt"
                        self.set_schwartz_probe_axis(value_a, value_b, checkpoint_path=checkpoint_path)
                    else:
                        print("Usage: probe_values VALUE_A VALUE_B [CHECKPOINT]")
                        print("  Example: probe_values SELF-DIRECTION SECURITY")
                        print("           probe_values ACHIEVEMENT BENEVOLENCE artifacts/my_probe.pt")
                    continue

                if user_input.lower().startswith('steer '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        behavior1 = parts[1]
                        behavior2 = parts[2]
                        try:
                            pooling = int(parts[3]) if len(parts) >= 4 else 1
                        except ValueError:
                            print("Usage: steer BEHAVIOR1 BEHAVIOR2 [POOLING]")
                            print("  POOLING must be -1, 1, or any positive integer k")
                            continue

                        if len(parts) < 4:
                            print("Using default pooling=1 (last token)")

                        self.set_steering_vector(behavior1, behavior2, pooling=pooling)
                    else:
                        print("Usage: steer BEHAVIOR1 BEHAVIOR2 [POOLING]")
                        print("  Default:  steer critical supportive      (pooling=1)")
                        print("  Examples: steer critical supportive -1")
                        print("            steer critical supportive 1")
                        print("            steer critical supportive 8")
                    continue

                print("\nGenerating response...")
                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _print_history(self):
        """Print the conversation history."""
        print("\n" + "=" * 60)
        print("Conversation History")
        print("=" * 60)
        for msg in self.conversation_history:
            print(f"{msg['role'].upper()}: {msg['content']}")
        print("=" * 60 + "\n")

    def _print_drift_scores(self, drift_scores: Dict[str, float]):
        """Print drift scores for the current interaction."""
        # Calculate statistics
        all_scores = list(drift_scores.values())
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)

        print("\n" + "-" * 60)
        print("Activation Drift Scores:")
        print(f"  Mean drift: {mean_score:.4f} (std: {std_score:.4f})")
        print(f"  Range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
        print("-" * 60 + "\n")

    def _resolve_layer_name(self, layer_spec: str, available_layers):
        """Resolve a layer spec to an exact layer key."""
        if layer_spec in available_layers:
            return layer_spec

        try:
            idx = int(layer_spec)
            candidate = f"model.layers.{idx}"
            if candidate in available_layers:
                return candidate
        except ValueError:
            pass

        return None

    def print_drift_evolution(self, layer: str = None):
        """Print how drift scores have evolved through the conversation.

        Args:
            layer: Optional layer spec (e.g., "20" or "model.layers.20").
                   If None, drift is averaged across layers.
        """
        if not self.drift_scores_history:
            print("No drift scores recorded yet")
            return

        if layer is not None:
            available_layers = self.drift_scores_history[0]["drift_scores"].keys()
            resolved_layer = self._resolve_layer_name(str(layer), available_layers)
            if resolved_layer is None:
                layer_list = sorted(available_layers, key=lambda x: int(x.split('.')[-1]))
                print(f"Layer '{layer}' not found")
                print(f"Available range: {layer_list[0]} ... {layer_list[-1]}")
                return
        else:
            resolved_layer = None

        print("\n" + "=" * 60)
        if resolved_layer is None:
            print("Activation Drift Evolution (mean across layers)")
        else:
            print(f"Activation Drift Evolution ({resolved_layer})")
        print("=" * 60)

        # Compute drift per turn (mean across layers or specific layer)
        for entry in self.drift_scores_history:
            turn = entry["turn"]
            scores = entry["drift_scores"]
            if resolved_layer is None:
                drift_value = np.mean(list(scores.values()))
            else:
                drift_value = scores[resolved_layer]

            user_input = entry["user_input"][:40]
            print(f"Turn {turn}: drift={drift_value:+.4f} | {user_input}...")

        print("=" * 60 + "\n")

        # Show overall trend
        if len(self.drift_scores_history) > 1:
            if resolved_layer is None:
                drifts = [np.mean(list(e["drift_scores"].values())) for e in self.drift_scores_history]
            else:
                drifts = [e["drift_scores"][resolved_layer] for e in self.drift_scores_history]
            initial_drift = drifts[0]
            final_drift = drifts[-1]
            change = final_drift - initial_drift

            print(f"Initial drift: {initial_drift:+.4f}")
            print(f"Final drift: {final_drift:+.4f}")
            print(f"Total change: {change:+.4f}")

            if change > 0.1:
                print(f"→ Drifting TOWARD {self.behavior1} (away from {self.behavior2})")
            elif change < -0.1:
                print(f"→ Drifting TOWARD {self.behavior2} (away from {self.behavior1})")
            else:
                print("→ Drift relatively stable")
            print()
