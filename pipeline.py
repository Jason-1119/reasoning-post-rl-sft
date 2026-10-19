"""Single-GPU teacher -> LoRA SFT -> LoRA GRPO; run each stage in a new process."""
import argparse
import hashlib
import json
import random
import re
import time
from fractions import Fraction
from pathlib import Path
from monitoring import Progress, training_callback

STUDENT = 'Qwen/Qwen3-1.7B-Base'
TEACHER = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
INSTRUCTION = ('Solve the problem step by step. End your response with a single line '
               'of the form #### <number>, containing only the final numeric answer.\n\n')


def prompt(question):
    # Plain text deliberately: the student is a Base model, not a chat model.
    return INSTRUCTION + 'Question:\n' + question.strip() + '\n\nSolution:\n'


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def numeric(text):
    """Conservative GSM8K-only parser. Never eval model-generated expressions."""
    text = str(text).strip().replace(',', '').replace('$', '')
    text = text.replace('−', '-').replace('\\,', '').strip()
    text = re.sub(r'\\(?:d?frac)\{([-+]?\d+)\}\{([-+]?\d+)\}', r'\1/\2', text)
    if len(text) > 100 or not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[-+]?\d+)?', text):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract(text):
    # Accept only an explicit final answer; no guessing from the last number.
    final = text.split('</think>')[-1]
    markers = re.findall(r'^\s*####\s*([^\n]+)\s*$', final, re.M)
    return markers[-1].strip() if markers else boxed(final)


def correct(text, gold):
    pred = extract(text)
    p, g = numeric(pred) if pred is not None else None, numeric(gold)
    return p is not None and g is not None and p == g


def normalize(question):
    return re.sub(r'\s+', '', question).casefold()


def uid(question):
    return hashlib.sha256(normalize(question).encode()).hexdigest()[:20]


def meta(root):
    return json.loads((root / 'manifest.json').read_text(encoding='utf-8'))


def prepare(a):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    revisions = {name: api.dataset_info(name).sha for name in
                 ['openai/gsm8k', 'HuggingFaceH4/MATH-500']}
    models = {name: api.model_info(name).sha for name in [STUDENT, TEACHER]}
    train = load_dataset('openai/gsm8k', 'main', split='train', revision=revisions['openai/gsm8k'])
    test = load_dataset('openai/gsm8k', 'main', split='test', revision=revisions['openai/gsm8k'])
    math = load_dataset('HuggingFaceH4/MATH-500', split='test', revision=revisions['HuggingFaceH4/MATH-500'])
    def gsm(row):
        return {'id': uid(row['question']), 'question': row['question'],
                'answer': row['answer'].split('####')[-1].strip(), 'dataset': 'gsm8k'}
    tests = [gsm(r) for r in test]
    maths = [{'id': uid(r['problem']), 'question': r['problem'], 'answer': r['answer'],
              'dataset': 'math500'} for r in math]
    forbidden = {normalize(r['question']) for r in tests + maths}
    seen, rows = set(), []
    for row in train:
        key = normalize(row['question'])
        if key not in seen and key not in forbidden:
            seen.add(key)
            rows.append(gsm(row))
    random.Random(a.seed).shuffle(rows)
    if a.test_n:
        # Fix the evaluation subset once, before any model is trained or evaluated.
        if a.test_n > len(tests):
            raise ValueError('test-n exceeds available GSM8K test rows')
        tests = random.Random(a.seed).sample(tests, a.test_n)
    if a.sft_n + a.rl_n + a.dev_n > len(rows):
        raise ValueError(f'Requested too many rows; available after dedup: {len(rows)}')
    splits = {'sft_questions': rows[:a.sft_n],
              'rl': rows[a.sft_n:a.sft_n+a.rl_n],
              'dev': rows[a.sft_n+a.rl_n:a.sft_n+a.rl_n+a.dev_n],
              'test_gsm8k': tests, 'test_math500': maths}
    for name, part in splits.items():
        write(a.root / 'data' / (name + '.jsonl'), part)
    manifest = {'seed': a.seed, 'dataset_revisions': revisions, 'model_revisions': models,
                'counts': {k: len(v) for k, v in splits.items()},
                'dedup': 'normalized exact matching only; does not remove semantic duplicates or pretraining overlap'}
    (a.root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def load_model(name, root, adapter=None, train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This profile requires a CUDA GPU with BF16 support. Run hardware check first.')
    revision = meta(root)['model_revisions'][name]
    print(f'[load] Loading {name} at revision {revision}. Download/load is separate from training progress.', flush=True)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa')
    model.config.pad_token_id = tok.pad_token_id
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    print(f'[load] Ready: {name}; GPU allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)
    return model, tok


def teacher(a):
    import torch
    from transformers import set_seed
    set_seed(a.seed)
    out = a.root / 'data' / 'teacher_attempts.jsonl'
    existing = read(out) if out.exists() else []
    done = {(r['id'], r['attempt']) for r in existing}
    rows = read(a.root / 'data' / 'sft_questions.jsonl')
    accepted_ids = {r['id'] for r in existing if r['accepted']}
    completed_ids = {r['id'] for r in rows if r['id'] in accepted_ids or
                     all((r['id'], i) in done for i in range(a.attempts))}
    progress = Progress(a.root, 'teacher', len(rows), initial=len(completed_ids))
    progress.update(len(completed_ids), accepted_questions=len(accepted_ids), attempts=len(existing))
    if len(completed_ids) == len(rows):
        print('[teacher] All questions already processed; skipping model loading.', flush=True)
        return
    model, tok = load_model(TEACHER, a.root)
    model.eval()
    # Append each attempt immediately: reruns skip completed (id, attempt) pairs.
    with out.open('a', encoding='utf-8') as f:
        for row in rows:
            if row['id'] in completed_ids:
                continue
            for attempt in range(a.attempts):
                if (row['id'], attempt) in done:
                    continue
                if any(r['id'] == row['id'] and r['accepted'] for r in existing):
                    break
                # No standard answer or reference solution is passed to the teacher.
                text = tok.apply_chat_template(
                    [{'role': 'user', 'content': INSTRUCTION + row['question']}],
                    tokenize=False, add_generation_prompt=True)
                if not text.endswith('<think>\n'):
                    text += '<think>\n'
                # Chat template already supplies BOS/special tokens (important for Llama).
                inputs = tok(text, add_special_tokens=False, return_tensors='pt').to(model.device)
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=a.teacher_tokens,
                                         do_sample=True, temperature=0.6, top_p=0.95,
                                         pad_token_id=tok.pad_token_id)
                new = ids[0, inputs.input_ids.shape[1]:]
                decoded = tok.decode(new, skip_special_tokens=True)
                eos = model.generation_config.eos_token_id or tok.eos_token_id
                eos = [eos] if isinstance(eos, int) else eos
                truncated = not (len(new) and int(new[-1]) in eos)
                record = {**row, 'attempt': attempt, 'text': decoded,
                          'tokens': len(new), 'seconds': time.perf_counter()-start,
                          'truncated': truncated,
                          'accepted': not truncated and correct(decoded, row['answer'])}
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
                f.flush()
                existing.append(record)
                if record['accepted']:
                    accepted_ids.add(row['id'])
                print(f"[teacher] question={row['id']} attempt={attempt+1}/{a.attempts} "
                      f"accepted={record['accepted']} tokens={len(new)} seconds={record['seconds']:.1f}", flush=True)
            completed_ids.add(row['id'])
            progress.update(len(completed_ids), accepted_questions=len(accepted_ids), attempts=len(existing))
    print(f'[teacher] Completed {len(rows)} questions; accepted={len(accepted_ids)}. '
          'Coverage after retries is not independent test accuracy.', flush=True)


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT, revision=meta(a.root)['model_revisions'][STUDENT])
    selected = {}
    for row in read(a.root / 'data' / 'teacher_attempts.jsonl'):
        if not row['accepted'] or not correct(row['text'], row['answer']):
            continue
        # Keep all reasoning, remove teacher wrappers and standardize only the answer marker.
        text = row['text'].replace('<think>', '').replace('</think>', '').strip()
        # Extracted teacher answer, never substitute the gold answer into an incorrect trace.
        completion = text + '\n#### ' + extract(row['text']) + tok.eos_token
        p = prompt(row['question'])
        length = len(tok(p + completion, add_special_tokens=False).input_ids)
        if length > a.seq_len:
            continue  # Never silently truncate the reasoning or final answer.
        candidate = {'id': row['id'], 'prompt': p, 'completion': completion, 'tokens': length}
        if row['id'] not in selected or length < selected[row['id']]['tokens']:
            selected[row['id']] = candidate
    write(a.root / 'data' / 'sft.jsonl', selected.values())
    print(f'{len(selected)} unique valid complete traces fit seq_len={a.seq_len}.')
    if not selected:
        raise RuntimeError('No valid traces. Inspect attempts; increase seq-len or generate shorter solvable problems.')


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias='none',
                      task_type='CAUSAL_LM', target_modules=[
                          'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])


def sft(a):
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(a.seed)
    model, tok = load_model(STUDENT, a.root)
    model = get_peft_model(model, lora_config())
    model.config.use_cache = False
    model.enable_input_require_grads()
    examples = []
    for row in read(a.root / 'data' / 'sft.jsonl'):
        p = tok(row['prompt'], add_special_tokens=False).input_ids
        c = tok(row['completion'], add_special_tokens=False).input_ids
        if len(p)+len(c) <= a.seq_len:
            examples.append({'input_ids': p+c, 'labels': [-100]*len(p)+c})
    if not examples:
        raise RuntimeError('Empty SFT dataset after length filtering.')
    import math
    expected_steps = a.steps if a.steps > 0 else math.ceil(len(examples)/8)
    print(f'[sft] Valid samples={len(examples)}; epochs=1; expected optimizer updates={expected_steps}', flush=True)
    def collate(batch):
        n = max(len(x['input_ids']) for x in batch)
        return {'input_ids': torch.tensor([x['input_ids']+[tok.pad_token_id]*(n-len(x['input_ids'])) for x in batch]),
                'attention_mask': torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in batch]),
                'labels': torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in batch])}
    args = TrainingArguments(
        output_dir=str(a.root/'sft'), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=1e-4, num_train_epochs=1,
        max_steps=a.steps, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1, save_steps=a.save_steps, save_total_limit=3,
        disable_tqdm=True, report_to='none', seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(examples),
                      data_collator=collate, processing_class=tok,
                      callbacks=[training_callback(a.root, 'sft')])
    model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(str(a.root/'sft'/'final'))
    tok.save_pretrained(a.root/'sft'/'final')
    print(f'[sft] Final adapter saved: {a.root / "sft" / "final"}', flush=True)


def reward(completions, answer, **kwargs):
    return [float(correct(c if isinstance(c, str) else c[0]['content'], g))
            for c, g in zip(completions, answer)]


def grpo(a):
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    set_seed(a.seed)
    adapter = a.root/'sft'/'final'
    if not adapter.exists():
        raise RuntimeError('SFT final adapter not found; run SFT first.')
    model, tok = load_model(STUDENT, a.root, adapter=adapter, train=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    tok.padding_side = 'left'
    rows = read(a.root/'data'/'rl.jsonl')
    print(f'[grpo] RL questions={len(rows)}; target updates={a.steps if a.steps > 0 else 64}; '
          '4 sampled completions per question; no KL penalty.', flush=True)
    ds = Dataset.from_list([{'prompt': prompt(r['question']), 'answer': r['answer']} for r in rows])
    args = GRPOConfig(
        output_dir=str(a.root/'grpo'), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_generations=4,
        max_prompt_length=1024, max_completion_length=a.rl_tokens,
        learning_rate=5e-6, max_steps=a.steps if a.steps > 0 else 64,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        temperature=0.8, top_p=0.95, beta=0.0, use_vllm=False,
        mask_truncated_completions=True, logging_steps=1,
        save_steps=a.save_steps, save_total_limit=3, disable_tqdm=True,
        report_to='none', seed=a.seed)
    # Continue the existing SFT adapter. No second adapter or base-model reset.
    # beta=0 deliberately: no KL reference model; monitor dev for drift.
    trainer = GRPOTrainer(model=model, processing_class=tok, args=args,
                          train_dataset=ds, reward_funcs=reward,
                          callbacks=[training_callback(a.root, 'grpo')])
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(str(a.root/'grpo'/'final'))
    tok.save_pretrained(a.root/'grpo'/'final')
    print(f'[grpo] Final adapter saved: {a.root / "grpo" / "final"}', flush=True)


def evaluate(a):
    import torch
    from transformers import set_seed
    set_seed(a.seed)
    adapter = None if a.stage == 'base' else a.root/a.stage/'final'
    model, tok = load_model(STUDENT, a.root, adapter=adapter)
    model.eval()
    rows = read(a.root/'data'/(a.split+'.jsonl'))
    if a.limit:
        rows = rows[:a.limit]
    math_task = a.split == 'test_math500'
    if math_task:
        from math_verify import parse, verify
    out = a.root/'eval'/f'{a.stage}_{a.split}_{a.eval_tokens}.jsonl'
    previous = read(out) if out.exists() else []
    done = {r['id'] for r in previous}
    progress = Progress(a.root, f'eval_{a.stage}_{a.split}', len(rows),
                        initial=sum(r['id'] in done for r in rows))
    completed = sum(r['id'] in done for r in rows)
    progress.update(completed)
    out.parent.mkdir(exist_ok=True)
    with out.open('a', encoding='utf-8') as f:
        for row in rows:
            if row['id'] in done:
                continue
            # Same text template and budget for all student stages, including MATH-500.
            p = prompt(row['question'])
            if math_task:
                p = p.replace('#### <number>, containing only the final numeric answer.',
                              '\\boxed{<answer>}, containing the final mathematical answer.')
            inputs = tok(p, add_special_tokens=False, return_tensors='pt').to(model.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=a.eval_tokens,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            new = ids[0, inputs.input_ids.shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            error = None
            if math_task:
                try:
                    ok = bool(verify(parse('\\boxed{'+str(row['answer'])+'}'), parse(text)))
                except Exception as exc:
                    ok, error = False, str(exc)
            else:
                ok = correct(text, row['answer'])
            result = {**row, 'text': text, 'correct': ok, 'tokens': len(new),
                      'answer_parseable': extract(text) is not None and numeric(extract(text)) is not None,
                      'seconds': time.perf_counter()-start, 'parse_error': error,
                      'budget_hit': len(new) >= a.eval_tokens and int(new[-1]) != tok.eos_token_id}
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            completed += 1
            progress.update(completed, correct=ok, tokens=len(new))
    chosen = {r['id'] for r in rows}
    results = [r for r in read(out) if r['id'] in chosen]
    n = len(results)
    if not n:
        raise RuntimeError('No evaluation rows.')
    lengths = sorted(r['tokens'] for r in results)
    import math
    successes = [r for r in results if r['correct']]
    summary = {'stage': a.stage, 'split': a.split, 'n': n,
               'correct_count': len(successes),
               'answer_parse_rate': (sum(r.get('answer_parseable', False) for r in results)/n
                                     if not math_task else None),
               'accuracy': len(successes)/n, 'eval_budget': a.eval_tokens,
               'mean_tokens': sum(lengths)/n, 'p95_tokens': lengths[math.ceil(.95*n)-1],
               'mean_correct_tokens': sum(r['tokens'] for r in successes)/len(successes) if successes else None,
               'mean_seconds': sum(r['seconds'] for r in results)/n,
               'budget_hit_rate': sum(r['budget_hit'] for r in results)/n,
               'parse_errors': sum(r['parse_error'] is not None for r in results)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'teacher', 'build', 'sft', 'grpo', 'eval'])
    p.add_argument('--root', type=Path, default=Path('runs/ds8b_qwen1p7b_128_001'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sft-n', type=int, default=128)
    p.add_argument('--rl-n', type=int, default=128)
    p.add_argument('--dev-n', type=int, default=32)
    p.add_argument('--test-n', type=int, default=64, help='Fixed GSM8K test subset; 0 means all test rows')
    p.add_argument('--attempts', type=int, default=2)
    p.add_argument('--teacher-tokens', type=int, default=4096)
    p.add_argument('--seq-len', type=int, default=4096)
    p.add_argument('--steps', type=int, default=-1)
    p.add_argument('--save-steps', type=int, default=16)
    p.add_argument('--rl-tokens', type=int, default=2048)
    p.add_argument('--eval-tokens', type=int, default=2048)
    p.add_argument('--resume', default='')
    p.add_argument('--stage', choices=['base', 'sft', 'grpo'], default='sft')
    p.add_argument('--split', choices=['dev', 'test_gsm8k', 'test_math500'], default='dev')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.command == 'prepare' and (a.root/'manifest.json').exists():
        p.error('Run already prepared. Use a new root to avoid changing splits underneath checkpoints.')
    if a.command != 'prepare':
        log = a.root/'commands.jsonl'
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(vars(a), default=str)+'\n')
    {'prepare': prepare, 'teacher': teacher, 'build': build,
     'sft': sft, 'grpo': grpo, 'eval': evaluate}[a.command](a)


if __name__ == '__main__':
    main()
