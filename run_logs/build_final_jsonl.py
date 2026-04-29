#!/usr/bin/env python3
"""Build the final research/MLINTERN_RESULTS.jsonl with val + test metrics
for every meaningful run.

Combines the per-run training metrics from extract_results.py with the
test-eval results in run_logs/phase4/.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path

EPOCH_PAT = re.compile(
    r'Epoch +(\d+) +\(([\d\.]+)s\) \[([\d\.]+)GB\] +'
    r'train\[vol=([\d\.\-]+) surf=([\d\.\-]+)\] +val_avg_surf_p=([\d\.]+)'
)
SPLIT_LINE = re.compile(
    r'    (val_\w+) +loss=([\d\.eE\+\-]+) +'
    r'surf\[p=([\d\.eE\+\-]+) Ux=([\d\.eE\+\-]+) Uy=([\d\.eE\+\-]+)\] +'
    r'vol\[p=([\d\.eE\+\-]+) Ux=([\d\.eE\+\-]+) Uy=([\d\.eE\+\-]+)\]'
)
EPOCH_HEADER = re.compile(r'^Epoch +(\d+) +\(([\d\.]+)s\)')

TEST_LINE = re.compile(
    r'    (test_\w+) +loss=([\d\.eE\+\-na]+) +'
    r'surf\[p=([\d\.eE\+\-na]+) Ux=([\d\.eE\+\-na]+) Uy=([\d\.eE\+\-na]+)\] +'
    r'vol\[p=([\d\.eE\+\-na]+) Ux=([\d\.eE\+\-na]+) Uy=([\d\.eE\+\-na]+)\]'
)
TEST_AVG = re.compile(r'TEST +avg_surf_p=([\d\.na]+)')


def parse_run(log: Path) -> dict | None:
    name = log.stem
    if name == 'queue':
        return None
    text = log.read_text(encoding='utf-8', errors='ignore')

    matches = EPOCH_PAT.findall(text)
    if not matches:
        return None

    epochs = []
    for m in matches:
        epochs.append({
            'epoch': int(m[0]),
            'epoch_time_s': float(m[1]),
            'peak_gb': float(m[2]),
            'train_vol_loss': float(m[3]),
            'train_surf_loss': float(m[4]),
            'val_avg_mae_surf_p': float(m[5]),
        })
    best = min(epochs, key=lambda e: e['val_avg_mae_surf_p'])
    last = epochs[-1]
    avg_dt = sum(e['epoch_time_s'] for e in epochs) / len(epochs)

    status = 'running'
    if 'Training done in' in text:
        status = 'done'
    if 'Timeout' in text:
        status = 'timeout'

    # Per-split metrics for the BEST epoch
    per_split: dict = {}
    matches_iter = list(EPOCH_HEADER.finditer(text))
    for i, m in enumerate(matches_iter):
        if int(m.group(1)) == best['epoch']:
            start = m.start()
            end = matches_iter[i + 1].start() if i + 1 < len(matches_iter) else len(text)
            chunk = text[start:end]
            for sm in SPLIT_LINE.finditer(chunk):
                split, loss, sp, sUx, sUy, vp, vUx, vUy = sm.groups()
                per_split[split] = {
                    'loss': float(loss),
                    'mae_surf_p': float(sp),
                    'mae_surf_Ux': float(sUx),
                    'mae_surf_Uy': float(sUy),
                    'mae_vol_p': float(vp),
                    'mae_vol_Ux': float(vUx),
                    'mae_vol_Uy': float(vUy),
                }
            break

    flags = {}
    cmd_lines = [l for l in text.splitlines() if 'train.py' in l and '--' in l][:1]
    if cmd_lines:
        line = cmd_lines[0]
        for fm in re.finditer(r'--(\w+)\s+(\S+)', line):
            flags[fm.group(1)] = fm.group(2)

    return {
        'name': name,
        'log': str(log),
        'status': status,
        'best_val_avg_mae_surf_p': best['val_avg_mae_surf_p'],
        'best_epoch': best['epoch'],
        'last_epoch': last['epoch'],
        'last_val_avg_mae_surf_p': last['val_avg_mae_surf_p'],
        'epoch_time_s_avg': avg_dt,
        'total_minutes': sum(e['epoch_time_s'] for e in epochs) / 60.0,
        'val_per_split_at_best_epoch': per_split,
        'config_flags': flags,
        'n_epochs_logged': len(epochs),
    }


def parse_test_log(log: Path) -> dict | None:
    text = log.read_text(encoding='utf-8', errors='ignore')
    per_split = {}
    for m in TEST_LINE.finditer(text):
        split, loss, sp, sUx, sUy, vp, vUx, vUy = m.groups()
        try:
            per_split[split] = {
                'loss': float(loss) if loss != 'nan' else None,
                'mae_surf_p': float(sp) if sp != 'nan' else None,
                'mae_surf_Ux': float(sUx) if sUx != 'nan' else None,
                'mae_surf_Uy': float(sUy) if sUy != 'nan' else None,
                'mae_vol_p': float(vp) if vp != 'nan' else None,
                'mae_vol_Ux': float(vUx) if vUx != 'nan' else None,
                'mae_vol_Uy': float(vUy) if vUy != 'nan' else None,
            }
        except ValueError:
            pass
    if not per_split:
        return None
    # Compute averages across the 4 splits.
    out = {'test_per_split': per_split}
    avg_keys = ['mae_surf_p', 'mae_surf_Ux', 'mae_surf_Uy',
                'mae_vol_p', 'mae_vol_Ux', 'mae_vol_Uy']
    for k in avg_keys:
        vals = [m[k] for m in per_split.values() if m.get(k) is not None]
        if len(vals) == len(per_split):
            out[f'test_avg/{k}'] = sum(vals) / len(vals)
    avg_match = TEST_AVG.search(text)
    if avg_match and avg_match.group(1) != 'nan':
        out['test_avg/mae_surf_p'] = float(avg_match.group(1))
    return out


def main():
    rows = []
    for d in ['run_logs/phase1', 'run_logs/phase2', 'run_logs/phase3', 'run_logs/phase4']:
        for log in sorted(Path(d).glob('*.log')):
            r = parse_run(log)
            if r:
                rows.append(r)

    # Match test eval logs to runs
    name_to_row = {r['name']: r for r in rows}
    for log in Path('run_logs/phase4').glob('test_eval_*.log'):
        run_name = log.stem.replace('test_eval_', '')
        # Skip the v1/amp variants we already replaced
        if run_name.endswith('-amp') or run_name == 'p3-amp-bs4':
            continue
        # Map test eval log back to its training run
        train_name = run_name
        if run_name.endswith('-v2'):
            train_name = run_name[:-len('-v2')] + '-warm-clip-180ep'
            train_name = 'p3-amp-bs4-warm-clip-180ep'
        test = parse_test_log(log)
        if not test:
            continue
        if train_name in name_to_row:
            name_to_row[train_name].update(test)

    rows.sort(key=lambda r: r['best_val_avg_mae_surf_p'])
    out_path = Path('research/MLINTERN_RESULTS.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    print(f'Wrote {len(rows)} runs to {out_path}\n')
    fmt = (
        f'{"name":<40}  {"val":>7}  {"test":>7}  '
        f'{"ep_b":>4}  {"ep_l":>4}  {"status":>8}  flags'
    )
    print(fmt)
    print('-' * len(fmt))
    for r in rows:
        flags_short = ' '.join(f'{k}={v}' for k, v in r['config_flags'].items()
                               if k not in ('agent', 'wandb_group', 'wandb_name', 'skip_test'))
        test = r.get('test_avg/mae_surf_p')
        test_s = f'{test:.2f}' if test is not None else '   —  '
        print(
            f"{r['name']:<40}  "
            f"{r['best_val_avg_mae_surf_p']:>7.2f}  "
            f"{test_s:>7}  "
            f"{r['best_epoch']:>4}  {r['last_epoch']:>4}  "
            f"{r['status']:>8}  {flags_short[:80]}"
        )


if __name__ == '__main__':
    main()
