# Qwen3-1.7B Math Reasoning: 128 Teacher Questions and 64 GRPO Updates

Teacher: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`  
Student: `Qwen/Qwen3-1.7B-Base`  
Pipeline: offline teacher generation -> verified traces -> one-epoch LoRA SFT -> 64 LoRA GRPO updates -> evaluation.

All project instructions, notebook cells, labels, comments, and runtime messages are in English. Models and datasets are downloaded at runtime; weights and generated data are not included.

## Default experiment

| Stage | Configuration |
|---|---|
| Teacher questions | 128 unique GSM8K training questions; up to two attempts each |
| SFT | One epoch on accepted complete traces; at most 128 samples |
| RL questions | 128 different GSM8K training questions |
| GRPO | 64 optimizer updates, four completions per question |
| Development set | 32 separate GSM8K training questions |
| Final test | 64 fixed randomly sampled GSM8K test questions, seed 42 |
| Optional evaluation | Teacher on the same 64 questions; MATH-500 available separately |

SFT, RL, development, and test questions are disjoint after normalized exact deduplication. Model and dataset revisions are saved in `manifest.json`. This does not eliminate semantic duplicates or pretraining contamination. With the same source revision and seed, the default GSM8K test subset matches the earlier 32-question project's test subset.

128 is the number of teacher candidate questions, not a guarantee of 128 accepted SFT samples. Incorrect, incomplete, unparseable, or overlength traces are rejected. Teacher prompts do not contain gold answers or reference solutions. Correct final answers do not certify every reasoning step; manually inspect a sample of accepted traces. The current recipe retains both reasoning and final explanations, which may contain repetition; that behavior is retained for comparability.

## Run in Colab

1. Extract this folder and upload it to `MyDrive/qwen3_1p7b_128_en`.
2. Open `Colab_Run.ipynb` in Colab and select a CUDA GPU with BF16 support.
3. Run the notebook cells in order. Restart the session if requested after dependency installation; do not delete the runtime.
4. Use the new default experiment directory `runs/ds8b_qwen1p7b_128_001`. Do not reuse old manifests, adapters, or evaluation outputs.

The notebook installs the fixed dependencies and repairs the specific Colab conflicts previously observed: missing Jedi, mismatched gcsfs/fsspec, and unused Gradio/Diffusers requiring a newer HF Hub. That repair removes Gradio/Diffusers from this temporary training environment; it does not delete data or model caches. Set `REPAIR_COLAB_PACKAGES=False` if your environment is already compatible. A successful `pip check` alone does not guarantee GPU-library compatibility; import checks and the first training updates still matter.

The profile targets the user's approximately 22.5 GB GPU. The teacher uses BF16, batch size one, and about 16 GB for weights alone; generation caches add memory. The student uses BF16 LoRA, not full-parameter training. Models run sequentially in separate subprocesses. Actual peak memory and speed must be measured on your runtime; GPU execution has not been tested locally.

## Progress you can see

Each subprocess streams output immediately to the Colab cell and writes a timestamped console log under `ROOT/logs/`. Generation and evaluation print question counts, completion percentage, elapsed time, and estimated remaining time. SFT and GRPO print progress after every optimizer update:

```text
[grpo] [##########----------] 32/64 (50.0%) elapsed=00:12:00 ETA=00:12:00 reward=0.5 ...
```

This is an illustrative format, not a measured training result or timing promise. The first RL update can take longer while four answers are generated; the display advances after an update, not after each token. Model downloads and loading have separate messages and are not optimizer updates. ETA is an estimate that becomes more useful after several updates and excludes earlier sessions when resuming.

| File | Contents |
|---|---|
| `logs/teacher_metrics.jsonl` | Completed teacher questions, accepted-question count, attempts, elapsed time, ETA |
| `logs/sft_metrics.jsonl` | SFT update number, loss, learning rate, epoch, GPU memory |
| `logs/grpo_metrics.jsonl` | GRPO update number and TRL logs, including reward, reward spread, completion lengths when provided |
| `logs/eval_*_metrics.jsonl` | Evaluation progress |
| `logs/*.console.log` | Full streamed stdout/stderr, including errors and checkpoints |
| `eval/comparison.csv` | Evaluation comparison exported by `inspect_run.py` |

GPU allocated/peak memory describe this training process, not all allocations on the GPU. Checkpoints are saved every 16 updates; the latest three are retained, and a final adapter is saved separately. SFT typically needs around 16 updates for 128 samples with gradient accumulation 8, and fewer if filtering removes samples. 64 GRPO updates is a fixed stopping point, not a claim of optimality.

The notebook includes result tables, progress inspection, and post-training loss/reward charts. You can inspect the saved state without loading a model:

```bash
python inspect_run.py --root runs/ds8b_qwen1p7b_128_001
```

## Command-line workflow

Run these commands from the project directory. Set `HF_HOME` to your preferred cache before starting a subprocess. The notebook automatically uses separate teacher/student caches.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest -v test_core.py

python pipeline.py prepare --root runs/ds8b_qwen1p7b_128_001
python pipeline.py teacher --root runs/ds8b_qwen1p7b_128_001
python pipeline.py build --root runs/ds8b_qwen1p7b_128_001
python pipeline.py sft --root runs/ds8b_qwen1p7b_128_001
python pipeline.py eval --root runs/ds8b_qwen1p7b_128_001 --stage sft --split dev
python pipeline.py grpo --root runs/ds8b_qwen1p7b_128_001 --steps 64
python pipeline.py eval --root runs/ds8b_qwen1p7b_128_001 --stage grpo --split dev

python pipeline.py eval --root runs/ds8b_qwen1p7b_128_001 --stage base --split test_gsm8k
python pipeline.py eval --root runs/ds8b_qwen1p7b_128_001 --stage sft --split test_gsm8k
python pipeline.py eval --root runs/ds8b_qwen1p7b_128_001 --stage grpo --split test_gsm8k
python inspect_run.py --root runs/ds8b_qwen1p7b_128_001
```

The GRPO CLI defaults to 64 updates even when `--steps` is omitted. For SFT, the default `--steps -1` means one epoch; a positive value overrides this with a fixed update count. All default generation budgets are unchanged: teacher 4096, SFT total sequence 4096, RL completion 2048, evaluation completion 2048.

## Resume and checkpoints

Teacher generation and evaluation save each answer and skip completed records on rerun. Training needs an actual Trainer checkpoint, including optimizer state:

```bash
python pipeline.py grpo --root runs/ds8b_qwen1p7b_128_001 --steps 64 --resume runs/ds8b_qwen1p7b_128_001/grpo/checkpoint-32
```

`--steps 64` is the total target, not 64 additional updates. Only use a checkpoint that exists. Work after the most recent checkpoint can be lost on interruption. Keep all data, prompts, dependencies, and training settings unchanged when resuming. The final adapter alone is sufficient for inference, but not exact optimizer-state resumption.

## Metrics and interpretation

- SFT learns only completion tokens; prompt/padding labels are masked.
- GRPO continues the SFT LoRA adapter with a correctness-only 0/1 reward. There is no length reward or KL penalty (`beta=0`).
- Under this single-GPU configuration, microbatch 1 x accumulation 4 gives four completions per optimizer update. With four generations per question, 64 updates corresponds to approximately 64 question-processing events and 256 completions, not 64 full passes over 128 questions.
- Groups with all-correct or all-wrong answers have little/no relative correctness learning signal. Look at reward and reward spread, not loss alone.
- Truncated RL completions are masked. A high truncation rate requires investigation before training longer.
- Compare Base/SFT/GRPO using the same test rows, prompt, budget, and greedy single-answer decoding.
- Accuracy is the strict verifier score. GSM8K requires an explicit `#### <number>` or numeric `\boxed{...}`; the verifier does not guess from the last number in prose. Answer parse rate is reported separately. Manually audit Base parse failures before interpreting low Base scores as lack of mathematical ability.
- A 64-question test changes by about 1.56 percentage points per answer. Use development data for tuning; viewing the test set repeatedly weakens its role as an untouched final evaluation.
- Teacher coverage after retries is not single-attempt test accuracy. Evaluate the teacher independently using `eval_teacher.py`.

Teacher evaluation uses its own chat template, the same saved test questions, the same 2048-token output budget, and the same final-answer verifier:

```bash
python eval_teacher.py --root runs/ds8b_qwen1p7b_128_001 --max-new-tokens 2048
```

Evaluation results support resuming. Do not overwrite trained adapters and reuse old evaluation filenames: use a new ROOT for a new experiment to avoid mixing old and new predictions. MATH-500 evaluation remains optional with `--split test_math500` and uses math-verify.

## Resource management

The notebook checks free disk space before downloading the teacher. It keeps the teacher cache by default so optional teacher evaluation does not redownload it. An explicit optional cell can remove only the project's known local teacher cache after data generation. No model caches are stored on Drive by default; datasets, logs, checkpoints, and reports are persisted there. Downloaded weights occupy disk; model weights and generation caches occupy GPU memory. Freeing disk does not increase VRAM.

If teacher generation runs out of GPU memory, stop it and reduce its generation limit only after inspecting truncation/acceptance effects. If student training runs out of memory, reduce sequence/completion lengths and re-filter SFT traces rather than silently truncating answers. Do not change just one GRPO batching parameter: generation batches must remain divisible by the number of generations. Exact runtime and compute-unit consumption depend on the actual GPU, output lengths, and download speed.

## Validation status

Local checks cover Python syntax, every notebook code cell, CLI entry points, numeric verification, answer extraction, reward interfaces, and progress persistence/resume ETA. No Colab dependency installation, model download, GPU training, or benchmark evaluation has been run in the local environment. See `VALIDATION.json` for check results. The project includes no trained weights or claimed benchmark gain.

## Official sources

- Teacher: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- Student: https://huggingface.co/Qwen/Qwen3-1.7B-Base
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- Pinned GRPO interface: https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer

Check model/dataset licenses and upstream terms before redistribution; the teacher also documents its Llama-3.1 origin and associated terms.
