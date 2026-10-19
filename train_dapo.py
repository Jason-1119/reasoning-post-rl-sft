"""Single-GPU DAPO experiment from an existing project SFT LoRA adapter.

Requires pipeline.py, monitoring.py, and exactly trl==0.23.1. No teacher needed.
This small-batch implementation includes bounded dynamic group sampling,
token-normalized DAPO loss, asymmetric clipping, and soft overlong rewards.
It is not an exact reproduction of the large-scale DAPO paper's setup.
"""
import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

from pipeline import STUDENT, correct, extract, numeric, prompt, read, load_model, meta
from monitoring import Progress, training_callback


def soft_penalty(length, limit, window, weight):
    return -weight * min(1., max(0., (length-(limit-window))/window))


def informative_group(correctness, active):
    values = [c for c, keep in zip(correctness, active) if keep]
    return len(values) >= 2 and min(values) < max(values)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(a):
    if not 0 < a.soft_window <= a.rl_tokens:
        raise ValueError('soft-window must be in (0, rl-tokens].')
    if a.iterations < 1 or a.max_resamples < 0 or a.penalty_weight < 0:
        raise ValueError('Invalid iterations, max-resamples, or penalty-weight.')
    if a.steps <= 0 or a.steps % a.iterations:
        raise ValueError('steps must be positive and divisible by iterations.')
    if not a.run_name or Path(a.run_name).name != a.run_name or a.run_name in {'.', '..'}:
        raise ValueError('run-name must be a simple directory name.')
    source = a.root/'sft'/'final'
    for path in [source/'adapter_config.json', source/'adapter_model.safetensors',
                 a.root/'data'/'rl.jsonl', a.root/'data'/'test_gsm8k.jsonl']:
        if not path.exists():
            raise FileNotFoundError(path)
    rl = read(a.root/'data'/'rl.jsonl')
    test = read(a.root/'data'/'test_gsm8k.jsonl')
    if not rl or not test:
        raise ValueError('RL and test sets must be nonempty.')
    if len({r['id'] for r in test}) != len(test):
        raise ValueError('Duplicate test IDs.')
    if {r['id'] for r in rl} & {r['id'] for r in test}:
        raise ValueError('RL/test overlap detected.')
    if importlib.metadata.version('trl') != '0.23.1':
        raise RuntimeError('This subclass is validated against TRL 0.23.1 source only. Install project requirements.')
    return source, rl, test


def settings(a, source):
    return {'algorithm': 'bounded-single-group-DAPO', 'student': STUDENT,
            'model_revision': meta(a.root)['model_revisions'][STUDENT],
            'sft_adapter_sha256': digest(source/'adapter_model.safetensors'),
            'sft_config_sha256': digest(source/'adapter_config.json'),
            'rl_sha256': digest(a.root/'data'/'rl.jsonl'),
            'test_sha256': digest(a.root/'data'/'test_gsm8k.jsonl'),
            'steps': a.steps, 'learning_rate': a.learning_rate,
            'rl_tokens': a.rl_tokens, 'eval_tokens': a.eval_tokens,
            'num_generations': 4, 'gradient_accumulation_steps': 4,
            'num_iterations': a.iterations, 'temperature': .8, 'top_p': .95,
            'epsilon': .2, 'epsilon_high': .28, 'beta': 0., 'loss_type': 'dapo',
            'soft_window': a.soft_window, 'penalty_weight': a.penalty_weight,
            'max_resamples': a.max_resamples, 'seed': a.seed, 'trl': '0.23.1'}


def train(a, source, rows, out):
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    set_seed(a.seed)
    stats = {'sampled_groups': 0, 'rejected_groups': 0, 'accepted_groups': 0,
             'generated_completions': 0, 'generated_tokens': 0}
    if a.resume:
        # Counts are an upper bound if unsaved work was sampled after the resumed checkpoint.
        old = out/'sampling_totals.json'
        if old.exists():
            stats.update(json.loads(old.read_text()))
        print('[dapo] Resume: sampling totals may include work after the resumed checkpoint.', flush=True)
    pool = [{'prompt': prompt(r['question']), 'answer': r['answer'], 'question_id': r['id']} for r in rows]
    latest = {}

    def dapo_reward(completions, answer, completion_ids, **kwargs):
        texts = [c if isinstance(c, str) else c[0]['content'] for c in completions]
        scores = [float(correct(text, gold)) for text, gold in zip(texts, answer)]
        lengths = [len(ids) for ids in completion_ids]
        penalties = [soft_penalty(n, a.rl_tokens, a.soft_window, a.penalty_weight) for n in lengths]
        latest.update(correctness=scores, lengths=lengths, penalties=penalties)
        return [score+penalty for score, penalty in zip(scores, penalties)]

    class DAPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            if self.accelerator.num_processes != 1 or len(inputs) != 4:
                raise RuntimeError('This dynamic sampler supports one GPU and exactly one group of four completions.')
            if not self.model.training:
                return super()._generate_and_score_completions(inputs)
            sampler = random.Random(a.seed+1009*self.state.global_step)
            candidates = list(range(len(pool)))
            sampler.shuffle(candidates)
            current = inputs
            for attempt in range(a.max_resamples+1):
                metrics_before = copy.deepcopy(self._metrics['train'])
                logs_before = copy.deepcopy(self._logs)
                token_count_before = self.state.num_input_tokens_seen
                result = super()._generate_and_score_completions(current)
                raw = latest['correctness']
                active = (result['completion_mask'].sum(dim=1) > 0).tolist()
                accepted = informative_group(raw, active)
                stats['sampled_groups'] += 1
                stats['generated_completions'] += len(raw)
                stats['generated_tokens'] += sum(latest['lengths'])
                stats['accepted_groups' if accepted else 'rejected_groups'] += 1
                event = {'optimizer_step_before_update': self.state.global_step,
                         'resample_index': attempt, 'accepted': accepted,
                         'question_id': current[0]['question_id'], 'raw_correctness': raw,
                         'lengths': latest['lengths'], 'soft_penalties': latest['penalties'],
                         'active_completions': active, 'timestamp': time.time(), **stats}
                with (out/'sampling.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(event)+'\n')
                (out/'sampling_totals.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
                print(f"[dapo sampling] step={self.state.global_step+1}/{a.steps} "
                      f"attempt={attempt+1}/{a.max_resamples+1} accepted={accepted} "
                      f"correct={sum(raw):.0f}/4 total_generated_tokens={stats['generated_tokens']}", flush=True)
                if accepted:
                    self._metrics['train']['dapo/raw_correctness'].append(sum(raw)/len(raw))
                    self._metrics['train']['dapo/soft_penalty'].append(sum(latest['penalties'])/len(raw))
                    self._metrics['train']['dapo/resamples'].append(attempt)
                    self._metrics['train']['dapo/rejection_fraction'].append(
                        stats['rejected_groups']/stats['sampled_groups'])
                    return result
                # Do not report discarded-group rewards as optimization-batch rewards.
                self._metrics['train'] = metrics_before
                self._logs = logs_before
                self.state.num_input_tokens_seen = token_count_before
                del result
                if attempt < a.max_resamples:
                    index = candidates[attempt % len(candidates)]
                    current = [dict(pool[index]) for _ in range(4)]
            raise RuntimeError(
                'Dynamic sampling exhausted its retry budget without a mixed-correctness active group. '
                'Training stopped; no zero-signal fallback update was performed. Inspect sampling.jsonl, '
                'truncation, and RL task difficulty. Resume from a saved checkpoint after investigating.')

    model, tok = load_model(STUDENT, a.root, adapter=source, train=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    tok.padding_side = 'left'
    config = GRPOConfig(
        output_dir=str(out/'checkpoints'), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, steps_per_generation=4, num_generations=4,
        num_iterations=a.iterations, max_prompt_length=1024, max_completion_length=a.rl_tokens,
        learning_rate=a.learning_rate, max_steps=a.steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        loss_type='dapo', epsilon=.2, epsilon_high=.28, beta=0., scale_rewards='group',
        temperature=.8, top_p=.95, mask_truncated_completions=True, use_vllm=False,
        logging_steps=1, save_steps=16, save_total_limit=3, report_to='none',
        disable_tqdm=True, seed=a.seed)
    (out/'grpo_config.json').write_text(config.to_json_string(), encoding='utf-8')
    print('[dapo] Starting from SFT, not the previous GRPO adapter.', flush=True)
    print(f'[dapo] {a.steps} optimizer updates; {a.iterations} updates per sampled group; '
          'dynamic sampling adds variable generation cost.', flush=True)
    trainer = DAPOTrainer(model=model, processing_class=tok, args=config,
                          train_dataset=Dataset.from_list(pool), reward_funcs=dapo_reward,
                          callbacks=[training_callback(out, 'dapo')])
    trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    trainer.save_model(str(out/'final'))
    tok.save_pretrained(out/'final')
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps({'global_step': trainer.state.global_step,
                                                        'timestamp': time.time()}), encoding='utf-8')
    del trainer, model, tok
    gc.collect()
    torch.cuda.empty_cache()


def summarize(rows, stage, budget):
    n = len(rows)
    parsed = [extract(r['text']) for r in rows]
    answers = [correct(r['text'], r['answer']) for r in rows]
    lengths = sorted(r['tokens'] for r in rows)
    return {'stage': stage, 'split': 'test_gsm8k', 'n': n, 'eval_budget': budget,
            'correct_count': sum(answers), 'accuracy': sum(answers)/n,
            'answer_parse_rate': sum(x is not None and numeric(x) is not None for x in parsed)/n,
            'mean_tokens': sum(lengths)/n, 'p95_tokens': lengths[math.ceil(.95*n)-1],
            'mean_seconds': sum(r['seconds'] for r in rows)/n,
            'budget_hit_rate': sum(r['budget_hit'] for r in rows)/n}


def evaluate(a, questions, out):
    import torch
    from transformers import set_seed
    set_seed(a.seed)
    if not (out/'TRAINING_COMPLETE.json').exists():
        raise RuntimeError('No completed DAPO run found. Finish training before evaluating final.')
    predictions = out/'test_predictions.jsonl'
    results = read(predictions) if predictions.exists() else []
    done = {r['id'] for r in results}
    progress = Progress(out, 'eval_dapo', len(questions), initial=len(done))
    progress.update(len(done))
    if len(done) < len(questions):
        model, tok = load_model(STUDENT, a.root, adapter=out/'final')
        model.eval()
        with predictions.open('a', encoding='utf-8') as f:
            for q in questions:
                if q['id'] in done:
                    continue
                inputs = tok(prompt(q['question']), add_special_tokens=False, return_tensors='pt').to(model.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**inputs, do_sample=False, max_new_tokens=a.eval_tokens,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                new = ids[0, inputs.input_ids.shape[1]:]
                text = tok.decode(new, skip_special_tokens=True)
                row = {**q, 'text': text, 'tokens': len(new), 'seconds': time.perf_counter()-start,
                       'correct': correct(text, q['answer']),
                       'budget_hit': len(new) >= a.eval_tokens and int(new[-1]) != tok.eos_token_id}
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
                f.flush()
                done.add(q['id'])
                results.append(row)
                progress.update(len(done), correct=row['correct'], tokens=row['tokens'])
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarize(results, 'dapo', a.eval_tokens)
    (out/'test_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    comparison = []
    gold = {q['id']: (q['question'], q['answer']) for q in questions}
    for stage in ['sft', 'grpo']:
        path = a.root/'eval'/f'{stage}_test_gsm8k_{a.eval_tokens}.jsonl'
        if path.exists():
            old = read(path)
            if len(old) == len(gold) and {r['id']: (r['question'], r['answer']) for r in old} == gold:
                comparison.append(summarize(old, stage, a.eval_tokens))
            else:
                print(f'[comparison] Skipping {stage}: saved test questions do not match.')
    comparison.append(summary)
    report = [{'stage': r['stage'], 'n': r['n'], 'correct': r['correct_count'],
               'accuracy_percent': round(100*r['accuracy'], 2),
               'answer_parse_percent': round(100*r['answer_parse_rate'], 2),
               'mean_tokens': round(r['mean_tokens'], 1), 'mean_seconds': round(r['mean_seconds'], 2),
               'budget_hit_percent': round(100*r['budget_hit_rate'], 2)} for r in comparison]
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print('\nFinal comparison (saved SFT/GRPO predictions, new DAPO predictions):', flush=True)
    for r in report:
        print(json.dumps(r), flush=True)
    print(f"\nDAPO test accuracy: {summary['correct_count']}/{summary['n']} = {100*summary['accuracy']:.2f}%")
    print('Artifacts:', out)
    print('Accuracy means exact-answer correctness, not classification precision.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Existing experiment containing sft/final and data/')
    p.add_argument('--run-name', default='dapo_001')
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=5e-6,
                   help='Match the previous GRPO baseline; use a different run-name for a LR ablation.')
    p.add_argument('--iterations', type=int, default=2,
                   help='Two policy updates per sampled group so clipping can affect reused rollouts.')
    p.add_argument('--rl-tokens', type=int, default=2048)
    p.add_argument('--eval-tokens', type=int, default=2048)
    p.add_argument('--soft-window', type=int, default=512)
    p.add_argument('--penalty-weight', type=float, default=1.)
    p.add_argument('--max-resamples', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=Path)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--train-only', action='store_true')
    a = p.parse_args()
    if a.eval_only and a.train_only:
        p.error('Choose at most one of eval-only and train-only.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Use one GPU; do not launch through multi-process accelerate.')
    source, rl, questions = validate(a)
    out = a.root/'dapo'/a.run_name
    if a.resume and not a.resume.resolve().is_relative_to((out/'checkpoints').resolve()):
        p.error('Resume only from this DAPO run, not an SFT or old GRPO checkpoint.')
    out.mkdir(parents=True, exist_ok=True)
    metadata = settings(a, source)
    config_path = out/'experiment.json'
    if config_path.exists() and json.loads(config_path.read_text()) != metadata:
        p.error('Saved run settings differ. Choose a new run-name.')
    config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if not a.eval_only:
        if (out/'TRAINING_COMPLETE.json').exists():
            p.error('Training already completed. Use --eval-only or a new run-name.')
        if any((out/'checkpoints').glob('checkpoint-*')) and not a.resume:
            p.error('Checkpoints already exist. Resume explicitly or use a new run-name.')
        if (out/'sampling.jsonl').exists() and not a.resume:
            p.error('Sampling from an earlier attempt exists. Use a new run-name or resume an existing checkpoint.')
        train(a, source, rl, out)
    if not a.train_only:
        evaluate(a, questions, out)


if __name__ == '__main__':
    main()
