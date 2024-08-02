from typing import List

from physioex.data.dataset import PhysioExDataset, DATASETS as _DATASETS
from physioex.data.datamodule import PhysioExDataModule


def get_datasets():
    return _DATASETS.keys()

def register_dataset(dataset: str = None, dataset_channels: List[str] = None):
    from loguru import logger
    
    if dataset in _DATASETS.keys():
        logger.info(f"Dataset {dataset} already registered")
        return dataset
    else:
        logger.info(f"Registering dataset {dataset}")
    
    _DATASETS[dataset] = dataset_channels
    
    return dataset
