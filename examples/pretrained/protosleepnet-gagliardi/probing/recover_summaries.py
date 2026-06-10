"""Recover summary.json files from per-fold metrics.json for stage_probe v3 runs.

The v3 jobs crashed after saving fold results due to a NameError (feat_dim).
This script reads the per-fold metrics and generates the missing summary.json.
"""
import glob
import json
import os
import sys

import numpy as np


def recover(base_dir):
    """Walk base_dir for task directories that have fold_*/metrics.json but no summary.json."""
    recovered = 0
    # Pattern: base_dir/config/model/source/task/fold_N/metrics.json
    # We want to find: base_dir/config/model/source/task/ directories
    task_dirs = set()
    for mpath in glob.glob(os.path.join(base_dir, "*/*/*/*/*/metrics.json")):
        task_dir = os.path.dirname(os.path.dirname(mpath))
        task_dirs.add(task_dir)

    for task_dir in sorted(task_dirs):
        summary_path = os.path.join(task_dir, "summary.json")
        if os.path.exists(summary_path):
            continue

        # Collect fold metrics
        fold_metrics = []
        fold_dirs = sorted(glob.glob(os.path.join(task_dir, "fold_*")))
        for fold_dir in fold_dirs:
            metrics_path = os.path.join(fold_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    fold_metrics.append(json.load(f))

        if not fold_metrics:
            continue

        # Build summary
        summary = {}
        for key in fold_metrics[0]:
            vals = [m[key] for m in fold_metrics]
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary["n_folds"] = len(fold_metrics)
        summary["type"] = "stage_conditioned"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        recovered += 1

    print(f"Recovered {recovered} summary.json files from {base_dir}")


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/sofia/scratch/pilot/pilot_2026_0042/WORK/probing_stage_v3"
    recover(base_dir)
