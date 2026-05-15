"""Extract contextualized embeddings for sjepa.

Loads the pretrained Signal-JEPA encoder, then extracts per-epoch
embeddings for every subject in the specified dataset(s) using
sliding-window encoding. Results are cached to disk. After extraction,
runs a 5-fold linear probe and saves results.

Usage:
    python examples/foundation/sjepa/extract_embeddings.py --gpu_id 0
    python examples/foundation/sjepa/extract_embeddings.py --gpu_id 0 --datasets sleepedf hmc
"""
import argparse

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models import SJEEncoder, extract_embeddings, linear_probe

MODEL_NAME = "sjepa"
L = 21


def main():
    parser = argparse.ArgumentParser(description=f"Extract embeddings for {MODEL_NAME}")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"

    dataset = SJEEncoder.get_dataset(args.datasets[0] if args.datasets else "hmc")
    model = SJEEncoder(in_chan=len(dataset.channels))

    print(f"Model: {type(model).__name__}, params={sum(p.numel() for p in model.parameters()):,}")

    cache_name = dataset.DATASET_NAME

    path = extract_embeddings(
        model=model,
        dataset=dataset,
        model_name=MODEL_NAME,
        dataset_name=cache_name,
        L=L,
        device=device,
        overwrite=args.overwrite,
    )
    print(f"Saved to {path}")

    linear_probe(
        model_name=MODEL_NAME,
        dataset_name=cache_name,
        device=device,
        upload=args.upload,
    )


if __name__ == "__main__":
    main()
