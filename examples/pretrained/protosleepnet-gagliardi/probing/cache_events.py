"""Extract per-subject metadata and per-epoch event labels into the embedding directory.

Saves alongside existing ``*_embeddings.npy`` and ``*_labels.npy``:

- ``{subject_id}_metadata.json`` — all subject-level metadata (age, sex, diagnosis, ...)
- ``{subject_id}_{event_type}.npy`` — binary (N_epochs,) int8 per event type
- ``{subject_id}_events_summary.json`` — aggregated event counts per subject
- ``manifest.json`` — directory-level summary

Usage:
    # sleepedf (metadata only, no events):
    python cache_events.py --dataset sleepedf --emb_dir .../sleepedf/all

    # SHHS visit1 with CVD outcomes:
    python cache_events.py --dataset shhs --visit 1 --emb_dir .../train \\
        --extra_csv /path/to/shhs-cvd-summary-dataset-0.21.0.csv \\
        --extra_csv_key nsrrid --extra_csv_filter visitnumber=1

    # Parkinsons night HOA:
    python cache_events.py --dataset parkinsons --recording night --group HOA \\
        --emb_dir .../parkinsons_night_HOA/all
"""
import argparse
import glob
import json
import os

import numpy as np
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.events import map_events_to_epochs

EVENT_TYPES = ["arousal", "respiratory", "desaturation", "limb_movement"]


def _json_serializable(v):
    """Convert numpy/non-serializable types for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def load_extra_csv(csv_path, key_column, filter_expr=None):
    """Load extra CSV and return dict keyed by key_column."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if filter_expr:
        for expr in filter_expr:
            col, val = expr.split("=", 1)
            col = col.strip()
            val = val.strip()
            # Try numeric
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            df = df[df[col] == val]

    result = {}
    for _, row in df.iterrows():
        key = str(int(row[key_column])) if isinstance(row[key_column], float) else str(row[key_column])
        result[key] = {
            col: _json_serializable(val) for col, val in row.items()
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata + event labels into embedding directory"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with *_embeddings.npy files")
    parser.add_argument("--channels", nargs="+", default=["EEG"],
                        help="Channels for dataset init (default: EEG)")
    # Dataset-specific parameters
    parser.add_argument("--visit", type=int, default=None)
    parser.add_argument("--cohort", type=int, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--recording", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    # Extra CSV for merging additional metadata (e.g., SHHS CVD)
    parser.add_argument("--extra_csv", type=str, default=None,
                        help="Path to extra CSV to merge into metadata")
    parser.add_argument("--extra_csv_key", type=str, default="nsrrid",
                        help="Key column in extra CSV")
    parser.add_argument("--extra_csv_filter", nargs="*", default=None,
                        help="Filter expressions like 'visitnumber=1'")
    args = parser.parse_args()

    # Build dataset kwargs
    ds_kwargs = {}
    if args.visit is not None:
        ds_kwargs["visit"] = args.visit
    if args.cohort is not None:
        ds_kwargs["cohort"] = args.cohort
    if args.subset is not None:
        ds_kwargs["subset"] = args.subset
    if args.recording is not None:
        ds_kwargs["recording"] = args.recording
    if args.group is not None:
        ds_kwargs["group"] = args.group

    # Scan embedding directory
    emb_files = sorted(glob.glob(os.path.join(args.emb_dir, "*_embeddings.npy")))
    if not emb_files:
        print(f"No *_embeddings.npy found in {args.emb_dir}")
        return
    emb_subject_ids = [
        os.path.basename(f).replace("_embeddings.npy", "") for f in emb_files
    ]
    print(f"Found {len(emb_subject_ids)} subjects in {args.emb_dir}")

    # Instantiate dataset (reads EDF headers + metadata CSVs, no signal loading)
    print(f"Loading dataset: {args.dataset} {ds_kwargs}")
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=args.channels,
        pipelines="seqsleepnet",
        sequence_length=0,
        **ds_kwargs,
    )
    ds_subjects = set(dataset.get_subjects())
    print(f"  Dataset has {len(ds_subjects)} subjects")

    # Load extra CSV if provided
    extra_meta = {}
    if args.extra_csv:
        print(f"Loading extra CSV: {args.extra_csv}")
        extra_meta = load_extra_csv(
            args.extra_csv, args.extra_csv_key, args.extra_csv_filter
        )
        print(f"  Extra CSV: {len(extra_meta)} entries")

    # Subject ID -> extra CSV key mapping functions
    # (needed to join embedding subject_ids with extra CSV keys)
    def _extract_nsrrid(sid):
        """Extract nsrrid from subject_id for NSRR datasets."""
        if "-" in sid:
            return sid.split("-", 1)[1]
        return sid

    # Check if dataset has events (try first subject)
    has_events = False
    for sid in list(ds_subjects)[:1]:
        try:
            evts = dataset.get_subject_events(sid)
            if evts:
                has_events = True
        except Exception:
            pass

    print(f"  Has events: {has_events}")

    # Process each subject
    n_metadata = 0
    n_events = 0
    n_skipped = 0
    n_missing = 0

    for subject_id in tqdm(emb_subject_ids, desc="subjects"):
        meta_path = os.path.join(args.emb_dir, f"{subject_id}_metadata.json")
        event_paths = {
            et: os.path.join(args.emb_dir, f"{subject_id}_{et}.npy")
            for et in EVENT_TYPES
        }
        summary_path = os.path.join(args.emb_dir, f"{subject_id}_events_summary.json")

        # Check if already done
        meta_exists = os.path.exists(meta_path)
        events_exist = has_events and all(os.path.exists(p) for p in event_paths.values())

        # If extra_csv is provided, re-check if metadata needs updating
        needs_meta_update = False
        if meta_exists and extra_meta:
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
                # Check if any extra CSV key is missing
                sample_key = next(iter(next(iter(extra_meta.values())).keys()))
                if sample_key not in existing:
                    needs_meta_update = True
            except Exception:
                needs_meta_update = True

        if meta_exists and not needs_meta_update and (not has_events or events_exist):
            n_skipped += 1
            continue

        if subject_id not in ds_subjects:
            n_missing += 1
            continue

        # --- Metadata ---
        if not meta_exists or needs_meta_update:
            try:
                meta = dataset.get_subject_metadata(subject_id)
                # Merge extra CSV data
                if extra_meta:
                    nsrrid = _extract_nsrrid(subject_id)
                    if nsrrid in extra_meta:
                        meta.update(extra_meta[nsrrid])
                # Make JSON-serializable
                meta = {k: _json_serializable(v) for k, v in meta.items()}
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=None, separators=(",", ":"))
                n_metadata += 1
            except Exception as e:
                print(f"  [META SKIP] {subject_id}: {e}")

        # --- Events ---
        if has_events and not events_exist:
            try:
                emb = np.load(
                    os.path.join(args.emb_dir, f"{subject_id}_embeddings.npy"),
                    mmap_mode="r",
                )
                n_epochs = emb.shape[0]

                events = dataset.get_subject_events(subject_id)
                epoch_events = map_events_to_epochs(events, n_epochs)

                summary = {"n_epochs": n_epochs}
                for et in EVENT_TYPES:
                    binary = np.array(
                        [
                            1 if any(e["type"] == et for e in epoch_evts) else 0
                            for epoch_evts in epoch_events
                        ],
                        dtype=np.int8,
                    )
                    np.save(event_paths[et], binary)
                    summary[et] = int(binary.sum())

                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=None, separators=(",", ":"))
                n_events += 1
            except Exception as e:
                print(f"  [EVENT SKIP] {subject_id}: {e}")

    # Save manifest
    manifest = {
        "dataset": args.dataset,
        "has_metadata": n_metadata > 0 or n_skipped > 0,
        "has_events": has_events,
        "event_types": EVENT_TYPES if has_events else [],
        "n_subjects": len(emb_subject_ids),
        "n_metadata_extracted": n_metadata,
        "n_events_extracted": n_events,
        "n_skipped": n_skipped,
        "n_missing_in_dataset": n_missing,
    }
    manifest_path = os.path.join(args.emb_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: metadata={n_metadata}, events={n_events}, "
          f"skipped={n_skipped}, missing={n_missing}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
