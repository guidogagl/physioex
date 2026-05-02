"""Extract per-epoch embeddings for chambon2018.

Uses direct per-epoch encoding (no sliding window) since Chambon2018's
epoch encoder has no inter-epoch context — each epoch is encoded
independently.  This is ~L times faster than the generic approach.

Usage:
    python examples/pretrained/chambon2018/extract_embeddings.py --gpu_id 0
    python examples/pretrained/chambon2018/extract_embeddings.py --gpu_id 0 --datasets mass hmc
"""
import argparse

import torch

from physioex.data.datasets import available_datasets, get_dataset
from physioex.data.presets import get_preset
from physioex.models.chambon2018 import Chambon2018Net, extract_embeddings
from physioex.models.embed import linear_probe
from physioex.models import load_from_pretrained

MODEL_NAME = "chambon2018"
CHANNELS = ["EEG"]
PIPELINE_PRESET = "raw"
PIPELINE_KWARGS = {"target_fs": 128.0}
SEQ_LEN = 3


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings for chambon2018"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--visit", type=int, default=None)
    parser.add_argument("--site", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cohort", type=int, default=None,
                        help="MASS cohort (1-5)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"

    model = load_from_pretrained(MODEL_NAME)

    # Initialize LazyLinear with a dummy forward
    with torch.no_grad():
        dummy = torch.randn(1, SEQ_LEN, 1, int(128.0 * 30))
        model(dummy)

    print(
        f"Model: {type(model).__name__}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    dataset_names = args.datasets if args.datasets else available_datasets()
    pipeline = get_preset(PIPELINE_PRESET, **PIPELINE_KWARGS)

    for ds_name in dataset_names:
        print(f"\nExtracting embeddings on {ds_name}...")
        try:
            DatasetClass = get_dataset(ds_name)
            ds_kwargs = dict(
                channels=CHANNELS,
                pipelines=pipeline,
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
            model_name=MODEL_NAME,
            dataset_name=cache_name,
            device=device,
            overwrite=args.overwrite,
        )
        print(f"  Saved to {path}")

        linear_probe(
            model_name=MODEL_NAME,
            dataset_name=cache_name,
            device=device,
            upload=args.upload,
        )


if __name__ == "__main__":
    main()
