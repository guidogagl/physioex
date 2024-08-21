import os
import numpy as np
import tempfile
import shutil
from tqdm import tqdm

def convert_npy_to_memmap(directory):
    # Esplora la directory e le sottodirectory
    for root, _, files in os.walk(directory):
        npy_files = [file for file in files if file.endswith('.npy')]
        if npy_files:
            print(f"Processing directory: {root}")
            for file in tqdm(npy_files, desc=f"Processing {root}", unit="file"):
                file_path = os.path.join(root, file)
                
                # Carica i dati dal file .npy
                data = np.load(file_path)
                
                # Crea un file temporaneo per il memmap
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file_name = tmp_file.name
                
                try:
                    # Crea il memmap e scrivi i dati
                    memmap = np.memmap(tmp_file_name, dtype=data.dtype, mode='w+', shape=data.shape)
                    memmap[:] = data[:]
                    memmap.flush()  # Assicurati che i dati siano scritti su disco
                    del memmap  # Rilascia il memmap
                    
                    # Sovrascrivi il file .npy originale con il memmap
                    shutil.move(tmp_file_name, file_path)
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    if os.path.exists(tmp_file_name):
                        os.remove(tmp_file_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Converti file .npy in memmap in una directory e sottodirectory.")
    parser.add_argument("--directory", type=str, help="La directory da esplorare.")
    
    args = parser.parse_args()
    
    convert_npy_to_memmap(args.directory)