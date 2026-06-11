"""Extract per-subject input spectrograms and save alongside embeddings.

For each subject, saves {sid}_inputs.npy with shape (N_epochs, 3, 29, 129)
into the same directory as the embeddings.

Usage:
    python extract_inputs.py \
        --dataset parkinsons --recording night --group HOA \
        --output_dir .../protosleepnet-st-3ch-mixer/parkinsons_night_HOA/all
"""
import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["parkinsons", "alzheimers"])
    parser.add_argument("--recording", default="night", choices=["night", "nap"])
    parser.add_argument("--group", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.dataset == "parkinsons":
        from physioex.data.datasets import ParkinsonsDataset
        ds = ParkinsonsDataset(
            recording=args.recording,
            group=args.group,
            channels=["EEG", "EOG", "EMG"],
            pipelines="seqsleepnet",
            sequence_length=0,
            cache_enabled=True,
        )
    elif args.dataset == "alzheimers":
        from physioex.data.datasets import AlzheimersDataset
        ds = AlzheimersDataset(
            group=args.group,
            channels=["EEG", "EOG", "EMG"],
            pipelines="seqsleepnet",
            sequence_length=0,
            cache_enabled=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    subjects = ds.get_subjects()
    print(f"Extracting {len(subjects)} subjects -> {args.output_dir}")

    for i in range(len(subjects)):
        batch = ds[i]
        sid = batch["subject"]["id"]
        channels = []
        for ch in sorted(batch["signals"].keys()):
            sig = batch["signals"][ch]
            if hasattr(sig, "numpy"):
                sig = sig.numpy()
            channels.append(sig)

        # (N, T, F) per channel -> stack to (N, C, T, F)
        inputs = np.stack(channels, axis=1).astype(np.float32)
        out_path = os.path.join(args.output_dir, f"{sid}_inputs.npy")
        np.save(out_path, inputs)
        print(f"  {sid}: {inputs.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
