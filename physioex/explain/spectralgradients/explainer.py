import math
import pickle
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from loguru import logger

from physioex.data import TimeDistributedModule
from physioex.explain.bands.importance import band_importance as compute_band_importance
from physioex.explain.bands.importance import eXpDataset
from physioex.explain.base import PhysioExplainer


def plot_class_importance(exp, band_names, class_names, filename):
    num_classes = len(class_names)
    num_cols = 2
    num_rows = math.ceil(num_classes / num_cols)

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(8.27, 11.69)
    )  # Dimensioni A4 in pollici
    axs = axs.flatten()

    # Creazione dei boxplot per ogni classe
    for i, class_name in enumerate(class_names):
        importance = exp.get_class_importance(i)
        class_df = pd.DataFrame(importance, columns=band_names)

        # Se class_df è vuoto, salta questa iterazione del ciclo
        if class_df.empty:
            print(f"No data for class {class_name}")
            continue

        # Creazione di un boxplot con un box per ogni banda
        melted_df = class_df.melt(var_name="Band", value_name="Importance")
        ax = sns.boxplot(x="Band", y="Importance", data=melted_df, ax=axs[i])
        # ax.set_ylim(0, 1)
        ax.set_title(f"Class Importance for {class_name}")

    for i in range(num_classes, num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_band_importance(exp, band_names, class_names, filename):
    num_classes = len(band_names)
    num_cols = 2
    num_rows = math.ceil(num_classes / num_cols)

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(8.27, 11.69)
    )  # Dimensioni A4 in pollici
    axs = axs.flatten()

    # Creazione dei boxplot per ogni classe
    for i, band_name in enumerate(band_names):
        importance = exp.get_bands_importance(i)
        band_df = pd.DataFrame(importance, columns=class_names)

        # Creazione di un boxplot con un box per ogni banda
        melted_df = band_df.melt(var_name="Class", value_name="Importance")
        ax = sns.boxplot(x="Class", y="Importance", data=melted_df, ax=axs[i])
        ax.set_title(f"Band Importance for {band_name}")

    for i in range(num_classes, num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class FreqBandsExplainer(PhysioExplainer):
    def __init__(
        self,
        model_name: str = "chambon2018",
        dataset_name: str = "sleep_physioex",
        loss_name: str = "cel",
        ckp_path: str = None,
        version: str = "2018",
        use_cache: bool = True,
        sequence_lenght: int = 3,
        batch_size: int = 32,
        sampling_rate: int = 100,
        class_name: list = ["Wake", "NREM1", "NREM2", "DeepSleep", "REM"],
    ):
        super().__init__(
            model_name,
            dataset_name,
            loss_name,
            ckp_path,
            version,
            use_cache,
            sequence_lenght,
            batch_size,
        )
        self.sampling_rate = sampling_rate
        self.class_name = class_name

    def compute_band_importance(
        self,
        bands: List[List[float]],
        band_names: List[str],
        fold: int = 0,
        plot_class: bool = False,
        save: bool = False,
    ):
        logger.info(
            "JOB:%d-Loading model %s from checkpoint %s"
            % (fold, str(self.model_call), self.checkpoints[fold])
        )
        model = self.model_call.load_from_checkpoint(
            self.checkpoints[fold], module_config=self.module_config
        ).eval()

        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = model.to(model_device)

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        self.dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=self.dataset,
            sequence_lenght=self.module_config["seq_len"],
            batch_size=self.batch_size,
            transform=self.input_transform,
            target_transform=self.target_transform,
        )

        self.module_config["loss_params"]["class_weights"] = datamodule.class_weights()

        dataloader = datamodule.train_dataloader()

        logger.info("JOB:%d-Computing bands importance" % fold)
        filename = self.ckpt_path + "explanations_fold_" + str(fold) + ".pt"

        explanations = compute_band_importance(
            bands,
            model,
            dataloader,
            self.sampling_rate,
        )

        model = model.cpu()

        logger.info("JOB:%d-Saving explanations" % fold)

        if save:
            with open(filename, "wb") as f:
                torch.save(explanations, f)

        logger.info("JOB:%d-Explanations saved" % fold)

        if plot_class:
            plot_class_importance(
                explanations,
                band_names,
                self.class_name,
                (self.ckpt_path + "fold=%d_class_importance.png") % fold,
            )

        return

    def explain(
        self,
        bands: List[List[float]],
        band_names: List[str],
        plot_class: bool = False,
        n_jobs: int = 10,
        save: bool = False,
    ):

        for fold in self.checkpoints.keys():
            self.compute_band_importance(bands, band_names, int(fold), plot_class, save)
        return


"""
# Execute compute_band_importance in parallel for every checkpoint
result = Parallel(n_jobs=n_jobs)(
    delayed(self.compute_band_importance)(
        bands, band_names, int(fold), plot_pred, plot_true,
    )
    for fold in self.checkpoints.keys()
)

return result
"""
