"""Console progress and durable JSONL metrics for generation and training."""
import json
import time
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return 'estimating'
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


class Progress:
    def __init__(self, root, stage, total, initial=0):
        self.stage, self.total = stage, total
        self.initial = initial
        self.started = time.monotonic()
        self.path = Path(root)/'logs'/f'{stage}_metrics.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, current, event='progress', **metrics):
        elapsed = time.monotonic()-self.started
        completed = current-self.initial
        eta = elapsed/completed * max(0, self.total-current) if completed > 0 else None
        record = {'stage': self.stage, 'event': event, 'current': current,
                  'total': self.total, 'elapsed_seconds': elapsed, 'eta_seconds': eta,
                  'timestamp': time.time(), **metrics}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str)+'\n')
        fraction = min(1., current/self.total) if self.total else 1.
        width = int(20*fraction)
        bar = '#' * width + '-' * (20-width)
        details = ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                           for k, v in metrics.items())
        print(f'[{self.stage}] [{bar}] {current}/{self.total} ({fraction:.1%}) '
              f'elapsed={duration(elapsed)} ETA={duration(eta)} {details}', flush=True)
        return record


def training_callback(root, stage):
    from transformers import TrainerCallback

    class TrainingMonitor(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.progress = Progress(root, stage, state.max_steps, initial=state.global_step)
            print(f'[{stage}] One progress step is one optimizer update. '
                  f'Target={state.max_steps}; resuming_at={state.global_step}.', flush=True)
            self.progress.update(state.global_step, event='train_begin')

        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = dict(logs or {})
            # Metrics include SFT loss / LR and TRL reward / reward_std / lengths.
            import torch
            if torch.cuda.is_available():
                metrics['gpu_allocated_gib'] = torch.cuda.memory_allocated()/2**30
                metrics['gpu_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
            self.progress.update(state.global_step, event='train_log', **metrics)

        def on_save(self, args, state, control, **kwargs):
            print(f'[{stage}] Checkpoint saved: {args.output_dir}/checkpoint-{state.global_step}', flush=True)

        def on_train_end(self, args, state, control, **kwargs):
            self.progress.update(state.global_step, event='train_end')

    return TrainingMonitor()
