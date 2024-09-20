from utils import set_root

from pytorch_lightning import seed_everything

seed_everything(42)
set_root()

from physioex.data import PhysioExDataModule
from physioex.models import load_pretrained_model
from physioex.models.load import get_models_table
from physioex.train.networks.utils.target_transform import get_mid_label

import pandas as pd
import numpy as np

import pytorch_lightning as pl


# iterate over the table rows and load the model corrispoding to the row

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default="/mnt/guido-data/")
parser.add_argument("--test_dataset", type=str, default="shhs")

args = parser.parse_args()

test_dataset = args.test_dataset
data_folder = args.data_folder

df = []

for i, row in get_models_table().iterrows():

    model = load_pretrained_model(
        name=row["name"],
        in_channels=row["in_channels"],
        sequence_length=row["sequence_length"],
    )

    for dataset in ["mros", "mesa", "dcsm", "hmc", "mass"]:

        data_module = PhysioExDataModule(
            datasets=[dataset],
            batch_size=256,
            preprocessing="xsleepnet" if row["name"] == "seqsleepnet" else "raw",
            selected_channels=(
                ["EEG"] if row["in_channels"] == 1 else ["EEG", "EOG", "EMG"]
            ),
            target_transform=None if row["name"] != "chambon2018" else get_mid_label,
            data_folder=data_folder,
        )

        trainer = pl.Trainer(deterministic=True)

        results = trainer.test(model, datamodule=data_module)[0]

        results["model"] = row["name"]
        results["in_channels"] = row["in_channels"]
        results["sequence_length"] = row["sequence_length"]
        results["dataset"] = dataset
        df.append(results)

df = pd.DataFrame(df)
df.to_csv("results.csv")
