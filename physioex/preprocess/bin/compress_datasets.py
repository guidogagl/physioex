import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def read_subject_record(data_folder, dataset_name, subject, num_windows, n_channels):
    subject = str(subject)

    # the values are stored as numpy memmaps
    raw = np.memmap(
        os.path.join(data_folder, dataset_name, "raw", f"{subject}.npy"),
        dtype="float32",
        mode="r",
        shape=(num_windows, n_channels, 3000),
    )[:]

    xsleep = np.memmap(
        os.path.join(data_folder, dataset_name, "xsleepnet", f"{subject}.npy"),
        dtype="float32",
        mode="r",
        shape=(num_windows, n_channels, 29, 129),
    )[:]

    labels = np.memmap(
        os.path.join(data_folder, dataset_name, "labels", f"{subject}.npy"),
        dtype="int16",
        mode="r",
        shape=(num_windows,),
    )[:]

    return subject, raw, xsleep, labels


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compress datasets")
    parser.add_argument(
        "--data_folder", default="/mnt/guido-data/", type=str, help="data folder"
    )
    parser.add_argument(
        "--datasets_name",
        default="mass hmc dcsm mros mesa shhs",
        type=str,
        help="datasets name",
    )
    parser.add_argument(
        "--output_folder", default="/mnt/guido-data/", type=str, help="output folder"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite the content of the output file"
    )
    parser.add_argument(
        "--n_jobs", default=1, type=int, help="number of jobs to run in parallel"
    )

    args = parser.parse_args()

    # convert the datasets name to a list
    datasets_name = args.datasets_name.split()

    # check that the datasets exist else raise a warning:
    for name in datasets_name:
        if not os.path.exists(os.path.join(args.data_folder, name)):
            print(f"Warning: {name} does not exist in {args.data_folder}")
            # remove the name from the list
            datasets_name.remove(name)

    # create the output folder if it does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    # process each dataset separately
    for name in datasets_name:
        print(f"Info: processing {name}")

        n_channels = 3 if name not in ["hmc", "dcsm"] else 4

        # create the output file for the current dataset
        h5_file_path = os.path.join(args.output_folder, f"{name}.h5")
        if os.path.exists(h5_file_path) and not args.force:
            print(f"Warning: {h5_file_path} already exists. Use --force to overwrite.")
            continue

        with h5py.File(h5_file_path, "w") as file:
            file.create_group("raw")
            file.create_group("xsleepnet")
            file.create_group("labels")

            # read the table of the dataset from the disk
            table = pd.read_csv(os.path.join(args.data_folder, name, "table.csv"))

            if "Unnamed: 0" in table.columns:
                table = table.drop("Unnamed: 0", axis=1)

            # for each column in the table create a dataset in the corresponding group
            for column in table.columns:
                if column in ["raw", "xsleepnet", "labels"]:  # string columns to skip
                    continue

                if (
                    "fold" in column
                ):  # the fold columns needs to be converted to int columns
                    table[column] = (
                        table[column]
                        .map({"train": 0, "valid": 1, "test": 2})
                        .astype(int)
                    )

                # create the dataset without compression
                file.create_dataset(
                    column,
                    data=np.reshape(table[column].astype(int).values, (-1)),
                    chunks=True,
                )

            # read and save the scaling information
            data = np.load(os.path.join(args.data_folder, name, "raw", "scaling.npz"))
            file["raw"].create_dataset("mean", data=data["mean"], chunks=True)
            file["raw"].create_dataset("std", data=data["std"], chunks=True)

            data = np.load(
                os.path.join(args.data_folder, name, "xsleepnet", "scaling.npz")
            )
            file["xsleepnet"].create_dataset("mean", data=data["mean"], chunks=True)
            file["xsleepnet"].create_dataset("std", data=data["std"], chunks=True)

            # read from the file the subject list
            subject_list = list(table["subject_id"])

            batch_size = args.n_jobs

            for i in tqdm(range(0, len(subject_list), batch_size)):
                batch_subjects = subject_list[i : i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(subject_list))))
                batch_num_windows = [table["num_windows"][idx] for idx in batch_indices]

                results = Parallel(n_jobs=args.n_jobs)(
                    delayed(read_subject_record)(
                        args.data_folder, name, subject, num_windows, n_channels
                    )
                    for subject, num_windows in zip(batch_subjects, batch_num_windows)
                )

                for subject, raw, xsleep, labels in results:
                    # create the datasets without compression
                    file["raw"].create_dataset(subject, data=raw, chunks=True)
                    file["xsleepnet"].create_dataset(subject, data=xsleep, chunks=True)
                    file["labels"].create_dataset(subject, data=labels, chunks=True)
