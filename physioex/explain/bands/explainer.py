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
from physioex.explain.bands.importance import \
    band_importance as compute_band_importance
from physioex.explain.bands.importance import eXpDataset as ExplanationsDataset
from physioex.explain.base import PhysioExplainer


def plot_band_importance(importance, y, band_names, class_names, filename):
    num_bands = len(band_names)
    num_cols = 2
    num_rows = math.ceil(num_bands / num_cols)

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(8.27, 11.69)
    )  # Dimensioni A4 in pollici
    axs = axs.flatten()  # Permette di iterare sugli assi in un unico ciclo

    for i, band in enumerate(band_names):
        df = pd.DataFrame(
            {
                "Band " + band_names[i] + " Importance": importance[:, i],
                "Class": y,
            }
        )

        ax = sns.boxplot(
            x="Class", y="Band " + band_names[i] + " Importance", data=df, ax=axs[i]
        )
        ax.set_xticklabels(class_names)
        ax.set_title("Band " + band_names[i] + " Importance for True Label")
        ax.set_xlabel("Class")
        ax.set_ylabel("Importance")

    # Rimuove gli assi vuoti se il numero di bande non è un multiplo del numero di colonne
    for i in range(num_bands, num_rows * num_cols):
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
        plot_pred: bool = False,
        plot_true: bool = False,
        compute_time: bool = True,
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
        explanations = compute_band_importance(
            bands, model, dataloader, self.sampling_rate, compute_time
        )

        logger.info("JOB:%d-Saving explanations" % fold)

        with open(
            self.ckpt_path + "explanations_fold_" + str(fold) + ".pkl", "wb"
        ) as f:
            pickle.dump(explanations, f, pickle.HIGHEST_PROTOCOL)

        logger.info("JOB:%d-Explanations saved" % fold)

        if plot_true:
            importance, y = explanations.get_true_importance()

            plot_band_importance(
                importance,
                y,
                band_names,
                self.class_name,
                (self.ckp_path + "fold=%d_true_band_importance.png") % fold,
            )

        if plot_pred:
            importance, y = explanations.get_pred_importance()

            plot_band_importance(
                importance,
                y,
                band_names,
                self.class_name,
                (self.ckp_path + "fold=%d_pred_band_importance.png") % fold,
            )

        return

    def explain(
        self,
        bands: List[List[float]],
        band_names: List[str],
        compute_time: bool = True,
        plot_pred: bool = False,
        plot_true: bool = False,
        n_jobs: int = 10,
    ):

        for fold in self.checkpoints.keys():
            self.compute_band_importance(
                bands, band_names, int(fold), plot_pred, plot_true, compute_time
            )
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
