"""Print saved progress and export a compact evaluation summary."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    print('Experiment:', a.root)
    manifest = a.root/'manifest.json'
    if manifest.exists():
        print('Dataset sizes:', json.loads(manifest.read_text())['counts'])
    for name in ['sft_questions', 'teacher_attempts', 'sft', 'rl', 'dev', 'test_gsm8k']:
        path = a.root/'data'/f'{name}.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as f:
                print(f'{name}: {sum(bool(line.strip()) for line in f)} records')
    print('\nLatest saved progress:')
    for path in sorted((a.root/'logs').glob('*_metrics.jsonl')):
        with path.open(encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        if lines:
            print(path.name, json.loads(lines[-1]))
    summaries = []
    for path in sorted((a.root/'eval').glob('*.summary.json')):
        r = json.loads(path.read_text())
        summaries.append({
            'stage': r['stage'], 'split': r['split'], 'n': r['n'],
            'accuracy_percent': round(100*r['accuracy'], 2),
            'answer_parse_percent': (round(100*r['answer_parse_rate'], 2)
                                     if r.get('answer_parse_rate') is not None else ''),
            'mean_tokens': round(r['mean_tokens'], 1),
            'mean_seconds': round(r['mean_seconds'], 2),
            'budget_hit_percent': round(100*r['budget_hit_rate'], 2),
            'eval_budget': r['eval_budget']})
    if summaries:
        out = a.root/'eval'/'comparison.csv'
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print('\nEvaluation results:')
        for row in summaries:
            print(row)
        print('CSV saved:', out)


if __name__ == '__main__':
    main()
