"""Extract contextualized embeddings for seqsleepnet-phan.

Loads the pretrained model from HuggingFace, then extracts per-epoch
embeddings for every subject in the specified dataset(s) using
sliding-window encoding.  Results are cached to disk.  After extraction,
runs a 5-fold linear probe and saves results.

Usage:
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0 --datasets sleepedf hmc
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0 --datasets shhs --visit 1 --dataset_root /data/shhs
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0 --datasets stages --site BOGN --dataset_root /data/stages
"""
import argparse

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models import extract_embeddings, linear_probe, load_from_pretrained

MODEL_NAME = "seqsleepnet-phan"
CHANNELS = ["EEG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 20


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings for seqsleepnet-phan"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--visit", type=int, default=None,
                        help="SHHS visit number (1 or 2)")
    parser.add_argument("--site", type=str, default=None,
                        help="STAGES site code (e.g. BOGN, GSBB)")
    parser.add_argument("--subset", type=str, default=None,
                        help="HPAP subset (lab-full, lab-split, home)")
    parser.add_argument("--cohort", type=int, default=None,
                        help="MASS cohort number (1-5)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"

    model = load_from_pretrained(MODEL_NAME)
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
            dataset = DatasetClass(**ds_kwargs)
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        if dataset.get_n_subjects() == 0:
            print(f"  [SKIP] {ds_name}: no subjects")
            continue

        # Use the dataset's own name for cache separation
        # (e.g. "shhs_visit1", "stages_BOGN" instead of "shhs", "stages")
        cache_name = dataset.DATASET_NAME

        path = extract_embeddings(
            model=model,
            dataset=dataset,
            model_name=MODEL_NAME,
            dataset_name=cache_name,
            L=SEQ_LEN,
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
