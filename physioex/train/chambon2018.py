import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from physioex.train.networks import *
from physioex.data import SleepPhysionet, Dreem, TimeDistributedModule

from pathlib import Path

import seaborn as sn
import pandas as pd

import torch
import numpy as np
from lightning.pytorch import seed_everything

from tqdm import tqdm

def get_mid_label( labels ):
    # labels is a seq_len vector

    return labels[ int((len(labels) - 1) / 2) ]

seed_everything(42, workers=True)
folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def chambon2018(similarity, dataset, version, use_cache, sequence_lenght, max_epoch, val_check_interval, batch_size ):
    Dataset = globals()[dataset]
    
    if similarity:
        ckp_path = "models/SCL/chambon2018/seqlen=" + str(sequence_lenght) + "/" + dataset + "/" + version + "/" 
    else:
        ckp_path = "models/CCL/chambon2018/seqlen=" + str(sequence_lenght) + "/" + dataset + "/" + version + "/"

    Path(ckp_path).mkdir(parents=True, exist_ok=True)

    for fold in folds:

        module_config["seq_len"] = sequence_lenght
        if similarity:
            module = ContrChambon2018Net( module_config = module_config ) 
        else:
            module = Chambon2018Net( module_config = module_config )

        dataset = Dataset(version=version, use_cache=use_cache)
        dataset.split( fold )

        datamodule = TimeDistributedModule(dataset = dataset, sequence_lenght = sequence_lenght, batch_size = batch_size, transform = None, target_transform = get_mid_label)

        # Definizione delle callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold
        )

        progress_bar_callback = RichProgressBar()

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=max_epoch,
            val_check_interval=val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True
        )

        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)
        val_results = trainer.test(ckpt_path="best", dataloaders=datamodule.val_dataloader())

        try:
            old_df = pd.read_csv(ckp_path + 'val_results.csv')
            if "Unnamed: 0" in old_df.columns:
                old_df.pop("Unnamed: 0")
        except:
            old_df = pd.DataFrame([])

        df = pd.DataFrame(val_results)
        df["fold"] = fold

        df = pd.concat([old_df, df])

        df.to_csv(ckp_path + 'val_results.csv')

        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        try:
            old_df = pd.read_csv(ckp_path + 'test_results.csv')
            if "Unnamed: 0" in old_df.columns:
                old_df.pop("Unnamed: 0")
        except:
            old_df = pd.DataFrame([])

        df = pd.DataFrame(test_results)
        df["fold"] = fold

        df = pd.concat([old_df, df])
        df.to_csv(ckp_path + 'test_results.csv')

