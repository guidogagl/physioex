"""Extract contextualized embeddings and run linear probing.

Loads a pretrained model from HuggingFace, then extracts per-epoch
embeddings for every subject in the specified dataset(s) using
sliding-window encoding. Runs linear probing (5-fold CV) on each dataset.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/extract_embeddings.py \
        --model_name sleeptransformer-gagliardi --repo_id 4rooms/sleep-prototypes --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/extract_embeddings.py \
        --model_name seqsleepnet-gagliardi --repo_id 4rooms/sleep-prototypes \
        --gpu_id 0 --datasets sleepedf hmc
    python examples/pretrained/protosleepnet-gagliardi/extract_embeddings.py \
        --model_name sleeptransformer-gagliardi --repo_id 4rooms/sleep-prototypes \
        --gpu_id 0 --datasets shhs --visit 2
    python examples/pretrained/protosleepnet-gagliardi/extract_embeddings.py \
        --model_name sleeptransformer-gagliardi --repo_id 4rooms/sleep-prototypes \
        --gpu_id 0 --datasets mass --cohort 3
"""
import argparse
import importlib
import json
import os

import torch

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models import extract_embeddings, linear_probe, load_from_pretrained

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings and run linear probing"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g. sleeptransformer-gagliardi)")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repo ID (default: 4rooms/physioex)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local model dir with model.pt + config.json (offline, skips HF)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save per-subject softmax predictions to linear_probe_predictions.json"
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--visit", type=int, default=None)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cohort", type=int, default=None,
                        help="MASS cohort (1-5)")
    parser.add_argument("--recording", type=str, default=None,
                        help="Parkinsons recording (night, nap)")
    parser.add_argument("--group", type=str, default=None,
                        help="Parkinsons group (HOA, PD)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"

    model = load_from_pretrained(args.model_name, repo_id=args.repo_id)
    print(
        f"Model: {type(model).__name__}, params={sum(p.numel() for p in model.parameters()):,}"
    )

    dataset_names = args.datasets if args.datasets else available_datasets()

    for ds_name in dataset_names:
        print(f"\nExtracting embeddings on {ds_name}...")
        try:
            DatasetClass = get_dataset(ds_name)
            ds_kwargs = dict(
                channels=CHANNELS,
                pipelines=PIPELINE,
                sequence_length=SEQ_LEN,
            )
            if args.dataset_root:
                ds_kwargs["root"] = args.dataset_root
            if args.visit is not None:
                ds_kwargs["visit"] = args.visit
            if args.site is not None:
                ds_kwargs["site"] = args.site
            if args.subset is not None:
                ds_kwargs["subset"] = args.subset
            if args.cohort is not None:
                ds_kwargs["cohort"] = args.cohort
            if args.recording is not None:
                ds_kwargs["recording"] = args.recording
            if args.group is not None:
                ds_kwargs["group"] = args.group
            dataset = DatasetClass(**ds_kwargs)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        if dataset.get_n_subjects() == 0:
            print(f"  [SKIP] {ds_name}: no subjects")
            continue

        cache_name = dataset.DATASET_NAME

        path = extract_embeddings(
            model=model,
            dataset=dataset,
            model_name=args.model_name,
            dataset_name=cache_name,
            L=SEQ_LEN,
            device=device,
            overwrite=args.overwrite,
        )
        print(f"  Saved to {path}")

        linear_probe(
            model_name=args.model_name,
            dataset_name=cache_name,
            device=device,
            upload=args.upload,
            save_predictions=args.save_predictions,
        )


if __name__ == "__main__":
    main()
