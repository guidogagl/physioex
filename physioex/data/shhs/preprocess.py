import os
import numpy as np
import scipy.io
import pandas as pd

from pathlib import Path

input_dir = str(Path().home()) + "/shhs/mat/"
output_dirs = {
    "raw": str(Path().home()) + "/shhs/raw/",
    "xsleepnet": str(Path().home()) + "/shhs/xsleepnet/",
}


class Statistics:
    def __init__(self, num_modalities, raw_shape=[3000], preprocessed_shape=[29, 129]):

        self.current_raw_mean = np.zeros((num_modalities, *raw_shape))
        self.current_raw_std = np.zeros_like(self.current_raw_mean)

        self.current_preprocessed_mean = np.zeros((num_modalities, *preprocessed_shape))
        self.current_preprocessed_std = np.zeros_like(self.current_preprocessed_mean)

        self.count = 0

    def add_values(self, raw_data, preprocessed_data):
        self.current_raw_mean += np.sum(raw_data, axis=0)
        self.current_raw_std += np.sum(np.square(raw_data), axis=0)

        self.current_preprocessed_mean += np.sum(preprocessed_data, axis=0)
        self.current_preprocessed_std += np.sum(np.square(preprocessed_data), axis=0)

        self.count += len(raw_data)

    def get(self):
        return (
            self.current_raw_mean / self.count,
            np.sqrt(
                self.current_raw_std / self.count
                - np.square(self.current_raw_mean / self.count)
            ),
            self.current_preprocessed_mean / self.count,
            np.sqrt(
                self.current_preprocessed_std / self.count
                - np.square(self.current_preprocessed_mean / self.count)
            ),
        )


def process_files(input_dir, output_dir_raw, output_dir_preprocessed, csv_path):
    subject_samples = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".mat"):
            subject_id, modality = (
                filename.split("_")[0][1:],
                filename.split("_")[1].split(".")[0],
            )
            data = scipy.io.loadmat(os.path.join(input_dir, filename))

            # todo: check this correctly
            raw_data = data["raw"]
            preprocessed_data = data["preprocessed"]

            new_filename = f"{int( subject_id )}_{modality}.mmap"

            # Create memmap for raw data
            raw_memmap = np.memmap(
                os.path.join(output_dir_raw, new_filename),
                dtype="float32",
                mode="w+",
                shape=raw_data.shape,
            )
            raw_memmap[:] = raw_data[:]
            raw_memmap.flush()
            del raw_memmap

            # Create memmap for preprocessed data
            preprocessed_memmap = np.memmap(
                os.path.join(output_dir_preprocessed, new_filename),
                dtype="float32",
                mode="w+",
                shape=preprocessed_data.shape,
            )
            preprocessed_memmap[:] = preprocessed_data[:]
            preprocessed_memmap.flush()
            del preprocessed_memmap

            # Calculate mean and std for raw data
            raw_mean = np.mean(raw_data)
            raw_std = np.std(raw_data)

            # Calculate mean and std for preprocessed data
            preprocessed_mean = np.mean(preprocessed_data)
            preprocessed_std = np.std(preprocessed_data)

            # Append to the list
            subject_samples.append(
                [
                    subject_id,
                    modality,
                    raw_data.shape[0],
                    raw_mean,
                    raw_std,
                    preprocessed_mean,
                    preprocessed_std,
                ]
            )

    # Create a DataFrame and save to csv
    df = pd.DataFrame(
        subject_samples,
        columns=[
            "Subject_ID",
            "Modality",
            "Samples",
            "Raw_Mean",
            "Raw_Std",
            "Preprocessed_Mean",
            "Preprocessed_Std",
        ],
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    # check if the output_dirs exists and create them if not
    for key in output_dirs.keys():
        Path(output_dirs[key]).mkdir(parents=True, exist_ok=True)
