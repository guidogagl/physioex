"""CLI entry point for evaluating a pretrained PhysioEx model."""
import argparse
import importlib
import os
import yaml

import pandas as pd

from physioex.data.dataset import PhysioExDataset
from physioex.train.trainer import Trainer
from physioex.train import stats as _stats
from physioex.train.logger import add_logger_cli_args, build_logger


def _import_class(spec):
    """Import a class from 'module.path:ClassName' spec."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def test_script():
    parser = argparse.ArgumentParser(
        description="Evaluate a PhysioEx model on datasets."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model class spec: 'module.path:ClassName'",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to pretrained checkpoint to evaluate",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, help="One or more dataset names"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (merged over defaults)",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON/YAML string with model constructor kwargs",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Directory to save results CSV",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--selected_channels", nargs="+", default=["EEG"])
    parser.add_argument("--seqlen", type=int, default=21)
    parser.add_argument("--preprocessing", type=str, default="raw")
    parser.add_argument(
        "--voting",
        action="store_true",
        help="Use sliding-window voting evaluation (recommended for full-night sequences)",
    )
    parser.add_argument(
        "--voting_L",
        type=int,
        default=21,
        help="Window length for voting evaluation",
    )
    parser.add_argument(
        "--per_subject",
        action="store_true",
        help="Report per-class metrics with mean/std/CI aggregated across subjects.",
    )
    add_logger_cli_args(parser)
    args = parser.parse_args()

    # Merge YAML config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        for k, v in config.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    # Parse model_kwargs string (supports JSON and YAML)
    try:
        import json

        model_kwargs = json.loads(args.model_kwargs)
    except Exception:
        model_kwargs = yaml.safe_load(args.model_kwargs) or {}

    # Build model and load checkpoint
    model_class = _import_class(args.model)
    model = model_class(**model_kwargs)
    model, _, _ = Trainer.load_checkpoint(model, args.ckpt_path)

    log_dir = args.log_dir or (
        os.path.join(args.results_path, "tb") if args.results_path else os.path.join(os.getcwd(), "tb")
    )
    logger = build_logger(
        args.logger,
        log_dir=log_dir,
        run_name=args.run_name,
        tags=args.tags,
    )

    results = []
    for ds_name in args.datasets:
        dataset = PhysioExDataset(
            datasets=[ds_name],
            selected_channels=args.selected_channels,
            seqlen=args.seqlen,
            preprocessing=args.preprocessing,
        )
        eval_kwargs = dict(
            model=model,
            dataset=dataset,
            fold=args.fold,
            gpu_id=args.gpu_id,
            per_subject=args.per_subject,
            ci_method=args.ci_method,
            n_bootstrap=args.n_bootstrap,
        )
        if args.voting:
            res = Trainer.voting_evaluate(L=args.voting_L, **eval_kwargs)
        else:
            res = Trainer.evaluate(**eval_kwargs)

        # Flatten scalar metrics (skip non-scalar like confusion_matrix, support).
        scalar = {k: v for k, v in res.items() if isinstance(v, (int, float))}
        scalar["dataset"] = ds_name
        scalar["fold"] = args.fold

        # Flatten per-subject aggregates into mean/std/ci columns.
        aggregated = res.get("aggregated")
        if aggregated:
            for metric_key, stat in aggregated.items():
                flat_key = metric_key.replace("/", "_")
                scalar[f"{flat_key}_mean"] = stat["mean"]
                scalar[f"{flat_key}_std"] = stat["std"]
                scalar[f"{flat_key}_ci_low"] = stat["ci_low"]
                scalar[f"{flat_key}_ci_high"] = stat["ci_high"]

        results.append(scalar)

        # Log the confusion matrix figure if a tracker is active.
        cm = res.get("confusion_matrix")
        if cm is not None:
            fig = _stats.figure_from_cm(cm)
            logger.log_figure(f"test/{ds_name}/confusion_matrix", fig, args.fold)
            import matplotlib.pyplot as plt

            plt.close(fig)

    logger.close()

    df = pd.DataFrame(results)
    if args.results_path:
        os.makedirs(args.results_path, exist_ok=True)
        out = os.path.join(args.results_path, "results.csv")
        df.to_csv(out, index=False)
        print(f"[Info] Results saved to {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    test_script()
