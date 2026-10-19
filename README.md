# reasoning-post-rl-sft

Controlled experiments comparing **RL followed by additional SFT** against **direct additional SFT** for Qwen3-1.7B reasoning models.

> Does an intermediate RL stage provide additional value when both training routes receive identical subsequent supervised fine-tuning?

This repository presents a small-scale experimental study. It does not introduce a new RL algorithm or claim a statistically established performance improvement.

## Overview

Starting from the same original SFT model, we compare:

```text
Route A: Original SFT → GRPO → Additional SFT
Route B: Original SFT ──────→ Additional SFT
```

Both routes receive the same additional supervised examples, learning rate, epoch count, batching configuration, and random seed.

The experiment reuses an existing RL checkpoint rather than rerunning RL. It evaluates four models:

| Model | Description |
|---|---|
| `original_sft` | Original SFT model |
| `original_rl` | Existing RL model derived from the original SFT model |
| `rl_then_sft` | RL model continued with additional SFT |
| `direct_sft` | Original SFT model continued with identical additional SFT |

**The additional SFT budget is matched, but total pipeline compute is not.** Route A includes the cost of the earlier RL stage.

## Models

- **Student:** [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- **Initial teacher:** [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- **Adaptation:** BF16 LoRA, rank 16
- **Training framework:** Transformers, PEFT, and TRL

The final model files are LoRA adapters. Inference requires the original base model and one selected adapter.

## Original Training

The source SFT model was trained for one epoch on verified teacher-generated reasoning traces from up to 128 GSM8K training questions.

The source GRPO configuration was:

| Parameter | Value |
|---|---:|
| RL training questions | 128 |
| Optimizer updates | 64 |
| Learning rate | `5e-6` |
| Completions per question | 4 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Maximum completion length | 2,048 |
| KL coefficient | 0 |
| Random seed | 42 |

The project uses `trl==0.23.1`, whose `GRPOConfig` defaults to DAPO-style loss normalization. The `grpo` label identifies the original training recipe; it does not imply use of the original sequence-normalized GRPO loss.

A separately saved DAPO adapter can also be selected as the RL source.

## Additional SFT Configuration

| Parameter | Both Routes |
|---|---|
| Training data | 64 new GSM8K training questions |
| Supervision | Original GSM8K reference solutions |
| Epochs | 1 |
| Learning rate | `1e-5` |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Optimizer updates in the reported run | 8 |
| Maximum sequence length | 2,048 |
| Precision | BF16 |
| Random seed | 42 |

Additional training examples are selected independently of model predictions. They are not mined from test-set errors.

Calculator annotation wrappers are removed from the GSM8K reference solutions while retaining the reasoning and final answers. Complete examples exceeding the sequence limit are excluded rather than truncated.

Both branches:

- Use identical pretokenized examples.
- Compute loss only on completion tokens, including EOS.
- Mask prompt and padding tokens.
- Start with fresh optimizers.
- Continue their respective source LoRA adapters.
- Save new adapters without overwriting the original models.

This is a **generic additional-SFT comparison**, not targeted regression repair.

## Evaluation

All four models are evaluated on the same 64 newly selected GSM8K test questions.

Questions recorded in the original project splits and saved external-evaluation subsets are excluded through normalized exact matching. This does not eliminate semantic duplicates or pretraining contamination.

Evaluation settings:

| Setting | Value |
|---|---|
| Dataset | GSM8K `main/test` |
| Questions | 64 |
| Decoding | Greedy |
| Answers per question | 1 |
| Maximum new tokens | 2,048 |
| Prompt | Identical plain-text student prompt |
| Grading | Strict numeric answer verification |

The verifier extracts an explicit `#### <number>` or numeric `\boxed{...}` answer. It does not guess the answer from the last number in the response.

Reported metrics include:

- Correct answer count and accuracy
- Answer parse rate
- Mean output length
- Mean generation time
- Generation-budget hit rate
- Paired correctness differences between the two routes

The final one-epoch adapters are evaluated. Test accuracy is not used to select checkpoints.

## Results

The following results were recorded from one completed Colab experiment using the saved **GRPO** adapter as the RL source.

| Model | Correct / 64 | Accuracy (%) | Parse Rate (%) | Mean Tokens | Mean Seconds | Budget Hit (%) |
|---|---:|---:|---:|---:|---:|---:|
| Original SFT | 45 | 70.31 | 95.31 | 290.66 | 21.66 | 3.12 |
| Original GRPO | 45 | 70.31 | 100.00 | 303.75 | 23.26 | 3.12 |
| GRPO → Additional SFT | **46** | **71.88** | 96.88 | 248.70 | 19.11 | 1.56 |
| Direct Additional SFT | 45 | 70.31 | 96.88 | **243.06** | **18.45** | 1.56 |

Both additional-SFT branches completed eight optimizer updates.

### Paired Comparison

| Outcome | Questions |
|---|---:|
| Both routes correct | 43 |
| Only RL → SFT correct | 3 |
| Only direct SFT correct | 2 |
| Both routes wrong | 16 |

### Findings

- Additional SFT shortened responses in both routes relative to their starting checkpoints.
- RL → SFT answered one more question correctly than direct additional SFT.
- Direct additional SFT produced the shortest responses and lowest mean generation time.
- The one-question accuracy difference does not establish a stable advantage for the intermediate RL stage.
- These results use a different test subset from earlier experiments, so absolute scores should not be directly compared across subsets.

The table is transcribed from the completed run. Independent verification requires the corresponding adapters, frozen questions, logs, environment metadata, and per-question predictions.

## Repository Structure

```text
reasoning-post-rl-sft/
├── README.md
├── RL_Then_SFT_vs_Direct_SFT.ipynb
├── compare_post_sft.py
├── pipeline.py
├── monitoring.py
├── requirements.txt
├── repair_colab.py
├── test_core.py
└── test_post_sft.py
```

The notebook embeds the comparison runner, so `compare_post_sft.py` is optional for notebook execution.

Model weights, datasets, and generated run directories are not included in the source repository.

## Run in Google Colab

1. Open `RL_Then_SFT_vs_Direct_SFT.ipynb` in Colab.
2. Select a CUDA GPU with BF16 support.
3. Mount the Drive containing the existing experiment.
4. Configure the project and model paths.
5. Run the notebook cells in order.

Example configuration:

```python
from pathlib import Path

PROJECT = Path(
    "/content/drive/MyDrive/reasoning-post-rl-sft"
)

ROOT = Path(
    "/content/drive/MyDrive/qwen3_1p7b_128_en/"
    "runs/ds8b_qwen1p7b_128_001"
)

RL_STAGE = "grpo"
DAPO_RUN = "dapo_001"
NAME = f"matched_{RL_STAGE}_post_sft_001"

TRAIN_N, TEST_N = 64, 64
EPOCHS = 1
LEARNING_RATE = 1e-5
```

`PROJECT` contains the shared scripts and requirements. `ROOT` points to the existing experiment containing the saved adapters, original data, and manifest.

To use the saved DAPO adapter:

```python
RL_STAGE = "dapo"
```

The original training notebook does not need to be open. Neither teacher generation nor the initial SFT/RL training is repeated.

Use a new experiment name when changing settings or source adapters.


## Limitations

- One training seed, 64 additional examples, and 64 test questions are insufficient for strong general conclusions.
- The RL source checkpoint is fixed; this experiment does not measure variability across independent RL runs.
- Only additional SFT compute is matched, not total pipeline compute.
- Additional supervision uses original GSM8K references rather than the teacher-generated traces used for initial SFT.
- Strict answer formatting can affect accuracy; parse failures should be inspected separately.
- Shorter outputs alone do not establish better reasoning.
- Exact deduplication cannot exclude semantic duplicates or pretraining contamination.
- Generation timing depends on hardware and runtime conditions.

## Related Work

- [OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles](https://arxiv.org/abs/2503.17352)
- [Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners](https://arxiv.org/abs/2510.04454)
- [Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR](https://arxiv.org/abs/2606.03087)

This project does not claim that alternating SFT and RL or adding post-RL supervision is novel.

Unlike OpenVLThinker, the current experiment does not generate next-round SFT data with an RL model or restart each iteration from a fixed original checkpoint. It continues saved adapters using shared reference-solution data.

## Reproducibility

Local checks cover Python/notebook syntax, command-line entry points, answer grading, progress logging, reference preprocessing, exact-overlap filtering, and paired comparison helpers.

The reported GPU results come from the completed Colab experiment and were not independently reproduced by the local development checks.

For verification, retain the experiment manifest, dependency and hardware records, source/final adapters, frozen test questions, training logs, and per-question predictions.
