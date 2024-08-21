import h5py
import numpy as np
import os
import io
import paramiko
import tempfile
from tqdm import tqdm
import pandas as pd

def write_subjet_data(h5_file, raw_signal, xsleepnet_signal, labels, subject_name):
    channels = ["EEG", "EOG", "EMG", "ECG"]
    channels = channels[:raw_signal.shape[1]]
    subject_group = h5_file.create_group(subject_name)
    raw = subject_group.create_group("raw")
    xsleepnet = subject_group.create_group("xsleepnet")
    
    for i, channel in enumerate(channels):
        raw_chunk_shape = (512 * 1024 // (3000 * 4), 3000)
        xsleepnet_chunk_shape = (512 * 1024 // (29 * 129 * 4), 29, 129)
        raw.create_dataset(channel, data=raw_signal[:, i], chunks=raw_chunk_shape, compression="gzip", compression_opts=9)
        xsleepnet.create_dataset(channel, data=xsleepnet_signal[:, i], chunks=xsleepnet_chunk_shape, compression="gzip", compression_opts=9)
    
    labels_chunk_shape = (min(512 * 1024 // labels[:].itemsize, labels.shape[0]))
    subject_group.create_dataset("labels", data=labels[:], chunks=labels_chunk_shape, compression="gzip", compression_opts=9)

def convert_to_h5py(dataset_folder: str, remote_folder: str = None, remote_user: str = None, remote_address: str = None):

    with h5py.File(os.path.join(dataset_folder, 'hpc_dataset.h5'), 'w') as h5_file:
        labels_folder = os.path.join(dataset_folder, 'labels')
        subject_index = []
        labels_files = [f for f in os.listdir(labels_folder) if f.endswith('.npy')]
        for subject_name in tqdm(labels_files, desc="Subject fetching"):
            labels = np.load(os.path.join(labels_folder, subject_name))
            raw = np.load(os.path.join(dataset_folder, 'raw', subject_name))
            xsleepnet = np.load(os.path.join(dataset_folder, 'xsleepnet', subject_name))
            write_subjet_data(h5_file, raw, xsleepnet, labels, subject_name)
            subject_index += np.arange(len(subject_index), len(subject_index) + labels.shape[0]).tolist()
        subject_index = np.array(subject_index).astype(np.uint32)
        h5_file.create_dataset("subject_index", data=subject_index, compression="gzip", compression_opts=9)
        
    if remote_folder and remote_user and remote_address:
        h5_buffer.seek(0)
        transport = paramiko.Transport((remote_address, 22))
        transport.connect(username=remote_user)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            with sftp.file(os.path.join(remote_folder, 'hpc_dataset.h5'), 'wb') as remote_file:
                total_size = h5_buffer.getbuffer().nbytes
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Trasferimento file .h5") as pbar:
                    while True:
                        chunk = h5_buffer.read(1024)
                        if not chunk:
                            break
                        remote_file.write(chunk)
                        pbar.update(len(chunk))
            table_path = os.path.join(dataset_folder, 'table.csv')
            if os.path.exists(table_path):
                with sftp.file(os.path.join(remote_folder, 'table.csv'), 'wb') as remote_file:
                    total_size = os.path.getsize(table_path)
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Trasferimento file table.csv") as pbar:
                        with open(table_path, 'rb') as local_file:
                            while True:
                                chunk = local_file.read(1024)
                                if not chunk:
                                    break
                                remote_file.write(chunk)
                                pbar.update(len(chunk))
        finally:
            sftp.close()
            transport.close()

if __name__ == "__main__":
    import argparse
    from physioex.utils import get_data_folder
    parser = argparse.ArgumentParser(description="Converti un dataset in formato HDF5")
    parser.add_argument("--dataset", type=str, help="Percorso della cartella del dataset")
    parser.add_argument("--remote_folder", type=str, default=None, help="Percorso della cartella remota")
    parser.add_argument("--remote_user", type=str, default=None, help="Nome utente per la connessione SSH")
    parser.add_argument("--remote_address", type=str, default=None, help="Indirizzo del server remoto")
    
    args = parser.parse_args()
    
    dataset_folder = os.path.join(get_data_folder(), args.dataset)
    
    convert_to_h5py(dataset_folder, args.remote_folder, args.remote_user, args.remote_address)