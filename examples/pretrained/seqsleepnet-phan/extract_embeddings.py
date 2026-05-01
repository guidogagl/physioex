"""Extract contextualized embeddings for seqsleepnet-phan.

Loads the pretrained model from HuggingFace, then extracts per-epoch
embeddings for every subject in the specified dataset(s) using
sliding-window encoding.  Results are cached to disk.

Usage:
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0
    python examples/pretrained/seqsleepnet-phan/extract_embeddings.py --gpu_id 0 --datasets sleepedf hmc
"""
import argparse

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models import extract_embeddings, load_from_pretrained

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
            dataset = DatasetClass(
                channels=CHANNELS,
                pipelines=PIPELINE,
                sequence_length=SEQ_LEN,
            )
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        if dataset.get_n_subjects() == 0:
            print(f"  [SKIP] {ds_name}: no subjects")
            continue

        path = extract_embeddings(
            model=model,
            dataset=dataset,
            model_name=MODEL_NAME,
            dataset_name=ds_name,
            L=SEQ_LEN,
            device=device,
            overwrite=args.overwrite,
            upload=args.upload,
        )
        print(f"  Saved to {path}")


if __name__ == "__main__":
    main()
