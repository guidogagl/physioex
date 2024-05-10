import os
import numpy as np
import h5py
import pandas as pd

from tqdm import tqdm

from pathlib import Path

input_dir = str(Path().home()) + "/shhs/mat/"
output_dirs = {
    "raw": str(Path().home()) + "/shhs/raw/",
    "xsleepnet": str(Path().home()) + "/shhs/xsleepnet/",
}

csv_path =  str(Path().home()) + "/shhs/table.csv"

modalities = ["eeg", "eog", "emg"]


class Statistics:
    def __init__(self, num_modalities, raw_shape=[3000], preprocessed_shape=[29, 129]):

        self.current_raw_mean = np.zeros((num_modalities, *raw_shape))
        self.current_raw_std = np.zeros_like(self.current_raw_mean)

        self.current_preprocessed_mean = np.zeros((num_modalities, *preprocessed_shape))
        self.current_preprocessed_std = np.zeros_like(self.current_preprocessed_mean)

        self.count = np.zeros(num_modalities)

    def add_values(self, modality_index, raw_data, preprocessed_data):
        self.current_raw_mean[modality_index] += np.sum(raw_data, axis=0)
        self.current_raw_std[modality_index] += np.sum(np.square(raw_data), axis=0)

        self.current_preprocessed_mean[modality_index] += np.sum(preprocessed_data, axis=0)
        self.current_preprocessed_std[modality_index] += np.sum(np.square(preprocessed_data), axis=0)

        self.count[modality_index] += len(raw_data)

    def get(self, modality_index):
        return (
            self.current_raw_mean[modality_index] / self.count[modality_index],
            np.sqrt(
                self.current_raw_std[modality_index] / self.count[modality_index]
                - np.square(self.current_raw_mean[modality_index] / self.count[modality_index])
            ),
            self.current_preprocessed_mean[modality_index] / self.count[modality_index],
            np.sqrt(
                self.current_preprocessed_std[modality_index] / self.count[modality_index]
                - np.square(self.current_preprocessed_mean[modality_index] / self.count[modality_index])
            ),
        )

def process_files(input_dir, output_dir_raw, output_dir_preprocessed, csv_path):
    subject_samples = []

    stats = Statistics(num_modalities = len(modalities) )
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".mat"):
            subject_id, modality = (
                filename.split("_")[0][1:],
                filename.split("_")[1].split(".")[0],
            )
            with h5py.File( input_dir + filename , 'r') as f:
                data = {key: f[key][()] for key in f.keys()}
            
            # todo: check this correctly
            raw_data = np.transpose( data["X1"], (1, 0) ).astype(np.float32)
            preprocessed_data = np.transpose( data["X2"], (2, 1, 0) ).astype(np.float32)
            
            if modality == "eeg":
                y = np.reshape( data["label"], (-1) ).astype(np.int16)
                fp = np.memmap(
                    os.path.join(output_dir_raw, f"y_{int(subject_id)}.mmap"),
                    dtype="float32",
                    mode="w+",
                    shape=y.shape,
                )
                fp[:] = y[:]
                fp.flush()
                del fp

                fp = np.memmap(
                    os.path.join(output_dir_preprocessed, f"y_{int(subject_id)}.mmap"),
                    dtype="float32",
                    mode="w+",
                    shape=y.shape,
                )
                fp[:] = y[:]
                fp.flush()
                del fp
                
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

            stats.add_values(modalities.index(modality), raw_data, preprocessed_data)
            
            # Append to the list
            subject_samples.append(
                [
                    subject_id,
                    modality,
                    raw_data.shape[0],
                ]
            )

    # Create a DataFrame and save to csv
    df = pd.DataFrame(
        subject_samples,
        columns=[
            "subject_id",
            "modality"
            "num_samples",
        ],
    )
    
    # remove from the df all the modalities wich are not EEG
    df = df[ df["modality"] == "eeg"] 
    df.drop( "modality" )
    
    df.to_csv(csv_path, index=False)

    raw_mean, raw_std, prepr_mean, prepr_std = [], [], [], []
    for modality in modalities:
        rm, rstd, pm, pstd = stats.get( modalities.index(modality) )
        
        raw_mean.append( np.array(rm).astype(np.float32) )
        raw_std.apend( np.array(rstd).astype(np.float32) )
        
        prepr_mean.append( np.array(pm).astype(np.float32) )
        prepr_std.apend( np.array(pstd).astype(np.float32) )
            
    np.savez(
        f"{output_dir_raw}/scaling.npz",
        mean = raw_mean,
        std = raw_std,
    )
    
    np.savez(
        f"{output_dir_preprocessed}/scaling.npz",
        mean = prepr_mean,
        std = prepr_std,
    )
    
if __name__ == "__main__":

    # check if the output_dirs exists and create them if not
    for key in output_dirs.keys():
        Path(output_dirs[key]).mkdir(parents=True, exist_ok=True)

    process_files(input_dir, output_dirs["raw"], output_dirs["xsleepnet"], csv_path)