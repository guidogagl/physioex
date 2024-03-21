import os
import wfdb

from loguru import logger

from physioex.data.base import PhysioExDataset
from physioex.data.utils import read_cache, write_cache
from pathlib import Path
import numpy as np 


def download_mit_bih():
    path = '~/mitdb/'
    path = os.path.expanduser(path)  # Espande il percorso '~' in un percorso assoluto

    # Controlla se il percorso esiste
    if not os.path.exists(path):
        logger.info("Creating new path {path} to store the dataset")
        os.makedirs(path)

    # Controlla se il dataset è già stato scaricato
    if not os.path.exists(os.path.join(path, '100.dat')):
        logger.info("Dataset not found, start fetching it from physionet")
        wfdb.dl_database('mitdb', path)
  # Ottiene un elenco di tutti i record nel database
    records = wfdb.get_record_list('mitdb')
    
    # Legge e restituisce tutti i record
    dataset = [wfdb.rdrecord(os.path.join(path, record)) for record in records]
    return dataset


class MITBIH(PhysioExDataset):
    def __init__(
        self,
        use_cache: bool = True,
        preprocessors=[
            lambda data: np.multiply(data, 1e6),  # Convert from V to uV
            lambda data: butter_bandpass_filter(data, 0.3, 30, 250),
            lambda data: resample(data, 100 * 30),
        ],
        picks=["C3_M2"],
    ):
    
    self.windows_dataset = None
    
    cache_path = "temp/mitdb.pkl"
    Path("temp/").mkdir(parents=True, exist_ok=True)

    if use_cache:
        self.windows_dataset = read_cache(cache_path)
        
    if self.windows_dataset:
            return

    dataset = download_mit_bih()  

    X = []
    y = []

    window_size = 100 * 30  # Ad esempio, per finestre di 30 secondi con campionamento a 100 Hz

    # Itera su ogni record nel dataset
    for record in dataset:
        # Applica il pre-processing e divide il record in finestre
        windows = np.array_split(record.p_signal, len(record.p_signal) // window_size)
        
        # Aggiunge le finestre al array X
        X.extend(windows)
        
        # Estrae l'etichetta dal commento del record e la aggiunge al array y per ogni finestra
        # Assumiamo che l'etichetta sia il primo commento di ogni record
        label = record.comments[0] if record.comments else None
        y.extend([label] * len(windows))

    # Converte le liste in array numpy
    X = np.array(X)
    y = np.array(y)
