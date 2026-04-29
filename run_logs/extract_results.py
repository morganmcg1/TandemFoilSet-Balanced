#!/usr/bin/env python3
"""Parse phase logs into a JSONL of per-run results.

For each completed/in-progress run we emit:
  {name, group, agent, status, best_val_avg_mae_surf_p, best_epoch,
   last_epoch, last_val_avg_mae_surf_p, per_split (best epoch),
   epoch_time_s_avg, total_minutes, config_flags, log_path}
"""
from __future__ import annotations

import argparse
import json
import re
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
    if 'Traceback' in text or 'CUDA out of memory' in text:
        status = 'error'

    # Extract per-split metrics for the BEST epoch.
    # The split lines follow each "Epoch N (...)\n" header sequentially.
    per_split = {}
    # Find the chunk for best epoch
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

    # Pull the launch command for config flags
    flags = {}
    cmd_lines = [l for l in text.splitlines() if 'train.py' in l and '--' in l][:1]
    if cmd_lines:
        # crude flag extraction
        line = cmd_lines[0]
        flag_iter = re.finditer(r'--(\w+)\s+(\S+)', line)
        for fm in flag_iter:
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
        'best_per_split': per_split,
        'config_flags': flags,
        'n_epochs_logged': len(epochs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='research/MLINTERN_RESULTS.jsonl')
    ap.add_argument('--logs', nargs='+', default=[
        'run_logs/phase1', 'run_logs/phase2', 'run_logs/phase3', 'run_logs/phase4',
    ])
    args = ap.parse_args()

    rows = []
    for d in args.logs:
        for log in sorted(Path(d).glob('*.log')):
            r = parse_run(log)
            if r:
                rows.append(r)

    rows.sort(key=lambda r: r['best_val_avg_mae_surf_p'])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    # Print compact ranking table
    print(f'Wrote {len(rows)} runs to {out}')
    print(f'{"val_avg/mae_surf_p":>20}  {"status":>8}  {"ep_best":>7}  {"ep_last":>7}  name')
    for r in rows:
        print(
            f"{r['best_val_avg_mae_surf_p']:>20.3f}  "
            f"{r['status']:>8}  "
            f"{r['best_epoch']:>7}  "
            f"{r['last_epoch']:>7}  "
            f"{r['name']}"
        )


if __name__ == '__main__':
    main()
