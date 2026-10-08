# Schwartz Probe Drift Pipeline

This guide covers the end-to-end workflow: extract labeled activations, train and validate a Schwartz probe, calibrate its logits, run a controlled conversation experiment, and inspect drift and conversation quality.

## Prerequisites

- Install the packages in `requirements.txt` in the Python environment used for the target model.
- Provide local copies of ValueNet, ValueEval, and FULCRA at the dataset paths shown below, or change the paths in the commands.
- Set `MODEL_PATH` to the local Hugging Face model directory used for both activation extraction and the experiment target. Use the same model and compatible tokenization throughout.
- The experiment's default simulated user uses Ollama Cloud. Configure Ollama access (including `OLLAMA_API_KEY` when using Ollama Cloud), or select the local simulator.

Replace checkpoint, model, and dataset paths with those for your environment.

## 1. Save Activations

Extract activations with Schwartz labels from the supported training datasets:

```bash
python process_schwartz.py all \
  --model-path "$MODEL_PATH" \
  --valuenet-dir data/ValueNet/v0.3_original \
  --valueeval-dir data/touche23_valueeval \
  --fulcra-dir data/value_fulcra \
  --valuenet-output activations/valuenet \
  --valueeval-output activations/valueeval \
  --fulcra-output activations/fulcra
```

Each example is saved with `activations.pkl` and `metadata.json`; the metadata carries the labels used by training. The processor saves last-token activations by default. To process one dataset, use `python process_schwartz.py valuenet`, `valueeval`, or `fulcra` with the corresponding `--dataset-dir` and `--output-dir` options.

## 2. Train the Probe

Train a single-layer probe. The checkpoint includes the split manifest used by evaluation and calibration:

```bash
python -m activation_drift.cli.train_schwartz_probe \
  --model-name llama-3.1-8b-instruct \
  --layer-name model.layers.31 \
  --pooling 1 \
  --valuenet-activations activations/valuenet \
  --valueeval-activations activations/valueeval \
  --fulcra-activations activations/fulcra \
  --output artifacts/schwartz_probe.pt
```

For layer selection, use `--layers 20 24 28 31` instead of `--layer-name`; training writes sweep summaries and one checkpoint per layer. See `python -m activation_drift.cli.train_schwartz_probe --help` for configuration-sweep options.

## 3. Evaluate the Probe

Evaluate the probe on the held-out test split.

```bash
python -m activation_drift.cli.eval_schwartz_probe \
  --checkpoint artifacts/schwartz_probe.pt \
  --valuenet-activations activations/valuenet \
  --valueeval-activations activations/valueeval \
  --fulcra-activations activations/fulcra \
  --split test \
  --output-dir artifacts/probe_eval
```

The evaluator writes `probe_eval_results.csv` and `probe_eval_results.json` under `artifacts/probe_eval`.

## 4. Calibrate the Probe

Estimate per-value logit means and standard deviations on the training split. Use the same activation roots used for training:

```bash
python -m activation_drift.cli.calibrate_schwartz_probe \
  --checkpoint artifacts/schwartz_probe.pt \
  --valuenet-activations activations/valuenet \
  --valueeval-activations activations/valueeval \
  --fulcra-activations activations/fulcra \
  --split train \
  --output artifacts/schwartz_probe_calibration_train.json
```

The calibration JSON is used to standardize probe logits, and is only used by the results processor. The experiment runner itself only requires the checkpoint, and returns raw logits scores.

## 5. Obtain Centroid Vectors for CAA or SoftCSA

[steering_vector_centroid.py](steering_vector_centroid.py) computes Schwartz value centroids from saved activations, for the CAA and SCAA methods. The experiment runner derives the pairwise steering vector from these centroids at runtime.

Use the trained probe checkpoint to inherit its layer, pooling, and train-split manifest:

```bash
python steering_vector_centroid.py \
  --checkpoint artifacts/schwartz_probe.pt \
  --split train \
  --valuenet-activations activations/valuenet \
  --valueeval-activations activations/valueeval \
  --fulcra-activations activations/fulcra \
  --output-dir steering_vectors_centroid
```

## Existing Artifacts

Pre-trained probes and activation centroids are available in the `probes/` and `centroids/` directories, respectively.


## 6. Run the Probe Experiment

Conversation templates define the scenarios and value pairs. Conditions define the model and user value targets; for example, `model_vb_user_va` steers the model toward value B and the user toward value A. The included files are [scenarios/scenarios_all.json](scenarios/scenarios_all.json) and [scenarios/conditions_all.json](scenarios/conditions_all.json).

```bash
python run_probe_drift_experiment.py \
  --target-model-path "$MODEL_PATH" \
  --checkpoint artifacts/schwartz_probe.pt \
  --conversations scenarios/scenarios_all.json \
  --conditions scenarios/conditions_all.json \
  --output-dir artifacts/probe_experiment_run \
  --num-seed-runs 5 
```

The default user simulator is cloud-based. To use a local simulator, add `--user-simulator local` and set `--user-role-model-path` to its model directory. For activation-based steering methods such as CAA or Soft CSA, provide the required steering vector options described by `--help`.

The run directory contains `runs.jsonl` (conversation records and traces), `probe_scores_by_turn.csv` (per-turn probe scores), and the experiment configuration. The default output directory, if omitted, is `artifacts/probe_experiment_<timestamp>`.

## 7. Evaluate Conversation Quality

This post-run analysis scores assistant responses for coherence and repetition. It is independent of the probe's held-out predictive evaluation. Coherence scoring uses the Ollama judge; repetition scoring uses the SentenceTransformers model.

```bash
python eval_conversation_quality.py \
  --result-dir artifacts/probe_experiment_run \
  --dump-file artifacts/probe_experiment_run/conversation_quality.pkl \
  --model llama8b \
  --normalization-json artifacts/schwartz_probe_calibration_train.json
```
## 8. Inspect and Plot Results

For normalized alignment metrics, aggregate plots, trajectories, conversation quality, and interactive transcript inspection, use [notebooks/process_results.ipynb](notebooks/process_results.ipynb). Further instructions are provided in the file.