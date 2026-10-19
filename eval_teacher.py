"""Evaluate the teacher on the saved student GSM8K test subset.

Compatible with the existing project's pipeline.py; no retraining required.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from pipeline import TEACHER, INSTRUCTION, load_model, read, correct, extract, numeric, meta
from monitoring import Progress


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--limit', type=int, default=0, help='0: entire saved subset, normally 64 questions')
    a = p.parse_args()
    import torch
    from transformers import set_seed
    set_seed(42)
    rows = read(a.root/'data'/'test_gsm8k.jsonl')
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise ValueError('Empty test subset')
    out = a.root/'eval'/f'teacher_test_gsm8k_{a.max_new_tokens}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        'teacher': TEACHER, 'revision': meta(a.root)['model_revisions'][TEACHER],
        'budget': a.max_new_tokens, 'decoding': 'greedy', 'prompt_version': 'teacher-user-only-think-v1',
        'test_sha256': hashlib.sha256((a.root/'data'/'test_gsm8k.jsonl').read_bytes()).hexdigest(),
    }
    settings_file = out.with_suffix('.settings.json')
    if settings_file.exists():
        if json.loads(settings_file.read_text()) != settings:
            raise ValueError('Existing result settings differ. Use a separate evaluation output file.')
    elif out.exists():
        raise ValueError('Existing teacher results have no settings metadata; rename them before running.')
    settings_file.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    done = {r['id'] for r in read(out)} if out.exists() else set()
    pending = [r for r in rows if r['id'] not in done]
    completed = len(rows)-len(pending)
    progress = Progress(a.root, 'eval_teacher_test_gsm8k', len(rows), initial=completed)
    progress.update(completed)
    if pending:
        model, tok = load_model(TEACHER, a.root)
        model.eval()
        eos = model.generation_config.eos_token_id or tok.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos
        with out.open('a', encoding='utf-8') as f:
            for row in pending:
                # Same user instruction as teacher data generation, no reference answer.
                text = tok.apply_chat_template(
                    [{'role': 'user', 'content': INSTRUCTION + row['question']}],
                    tokenize=False, add_generation_prompt=True)
                if not text.endswith('<think>\n'):
                    text += '<think>\n'
                inputs = tok(text, add_special_tokens=False, return_tensors='pt').to(model.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens,
                                         pad_token_id=tok.pad_token_id, eos_token_id=eos)
                torch.cuda.synchronize()
                new = ids[0, inputs.input_ids.shape[1]:]
                response = tok.decode(new, skip_special_tokens=True)
                answer = extract(response)
                result = {**row, 'text': response, 'correct': correct(response, row['answer']),
                          'tokens': len(new), 'seconds': time.perf_counter()-start,
                          'parse_error': None,
                          'answer_parseable': answer is not None and numeric(answer) is not None,
                          'budget_hit': len(new) >= a.max_new_tokens and int(new[-1]) not in eos_ids}
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
                f.flush()
                completed += 1
                progress.update(completed, correct=result['correct'], tokens=len(new))
    chosen = {r['id'] for r in rows}
    results = [r for r in read(out) if r['id'] in chosen]
    n = len(results)
    successes = [r for r in results if r['correct']]
    lengths = sorted(r['tokens'] for r in results)
    summary = {'stage': 'teacher', 'model': TEACHER, 'split': 'test_gsm8k', 'n': n,
               'accuracy': len(successes)/n, 'correct_count': len(successes),
               'eval_budget': a.max_new_tokens, 'mean_tokens': sum(lengths)/n,
               'mean_correct_tokens': sum(r['tokens'] for r in successes)/len(successes) if successes else None,
               'p95_tokens': lengths[math.ceil(.95*n)-1],
               'mean_seconds': sum(r['seconds'] for r in results)/n,
               'budget_hit_rate': sum(r['budget_hit'] for r in results)/n,
               'answer_parse_rate': sum(r['answer_parseable'] for r in results)/n,
               'parse_errors': 0}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
