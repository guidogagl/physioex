import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from loguru import logger
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from physioex.data import TimeDistributedModule, datasets
from physioex.explain.base import PhysioExplainer
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config

torch.set_float32_matmul_precision("medium")


def compute_projections(model, dataloader, model_device):
    train_projections = []
    y_train_true = []
    y_train_pred = []

    for batch in dataloader:
        inputs, y_true = batch

        y_train_true.append(y_true)

        projections, y_pred = model.encode(inputs.to(model_device))

        y_train_pred.append(y_pred.cpu().detach().numpy())
        train_projections.append(projections.cpu().detach().numpy())

        del projections, y_pred

    y_train_true = np.concatenate(y_train_true).reshape(-1)
    train_projections = np.concatenate(train_projections).reshape(
        y_train_true.shape[0], -1
    )
    y_train_pred = np.argmax(np.concatenate(y_train_pred).reshape(-1, 5), axis=1)

    return train_projections, y_train_true, y_train_pred


class ARIExplainer(PhysioExplainer):
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
    ):

        assert ckp_path is not None, "ckp_path must be provided"
        assert os.path.isdir(
            ckp_path
        ), "ckp_path must be a valid directory containing at least one checkpoint"

        self.model_name = model_name
        self.model_call = config[model_name]["module"]
        self.input_transform = config[model_name]["input_transform"]
        self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = sequence_lenght

        self.module_config["loss_call"] = loss_config[loss_name]
        self.module_config["loss_params"] = dict()

        self.batch_size = batch_size
        self.version = version
        self.use_cache = use_cache

        logger.info("Scanning checkpoint directory...")
        self.checkpoints = {}
        for elem in os.scandir(ckp_path):
            if elem.is_file() and elem.name.endswith(".ckpt"):
                try:
                    fold = int(re.search(r"fold=(\d+)", elem.name).group(1))
                except Exception as e:
                    logger.warning(
                        "Could not parse fold number from checkpoint name: %s. Skipping..."
                        % elem.name
                    )
                    continue
                self.checkpoints[fold] = elem.path

        logger.info("Found %d checkpoints" % len(self.checkpoints))
        self.ckpt_path = ckp_path

        logger.info("Loading dataset")
        self.dataset_call = datasets[dataset_name]
        self.dataset = self.dataset_call(version=self.version, use_cache=self.use_cache)
        logger.info("Dataset loaded")

    def compute_ari(
        self, fold: int = 0, plot_pca: bool = False, plot_kmeans: bool = False
    ):
        logger.info(
            "JOB:%d-Loading model %s from checkpoint %s"
            % (fold, str(self.model_call), self.checkpoints[fold])
        )
        model = self.model_call.load_from_checkpoint(
            self.checkpoints[fold], module_config=self.module_config
        ).eval()

        model_device = next(model.parameters()).device

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

        projections, y_true, y_pred = compute_projections(
            model, datamodule.train_dataloader(), model_device
        )

        if plot_pca:
            logger.info("JOB:%d-Computing PCA" % fold)
            pca = PCA(n_components=2)
            components = pca.fit_transform(projections)

            logger.info("JOB:%d-Plotting PCA" % fold)
            plt.figure(figsize=(10, 10))
            sns.scatterplot(
                x=components[:, 0], y=components[:, 1], hue=y_pred, palette="Set1"
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(self.model_name + " PCA")
            plt.tight_layout()
            plt.savefig(self.ckpt_path + "fold=%d_pca.png" % fold)
            plt.close()

        if plot_kmeans:
            logger.info("JOB:%d-Computing KMeans" % fold)
            kmeans = KMeans(
                n_clusters=self.module_config["n_classes"], random_state=0
            ).fit(projections)

            logger.info("JOB:%d-Computing PCA" % fold)
            pca = PCA(n_components=2)
            components = pca.fit_transform(projections)

            logger.info("JOB:%d-Plotting PCA" % fold)
            plt.figure(figsize=(10, 10))
            sns.scatterplot(
                x=components[:, 0],
                y=components[:, 1],
                hue=kmeans.labels_,
                palette="Set1",
            )
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(self.model_name + " PCA")
            plt.tight_layout()
            plt.savefig(self.ckpt_path + "fold=%d_kmeans_pca.png" % fold)
            plt.close()

        logger.info("JOB:%d-Computing ARI" % fold)
        tested_k = int(self.module_config["n_classes"] * 2)
        tested_k = list(range(1, tested_k + 1))

        ari_values = []

        for _, k in tqdm(enumerate(tested_k)):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(projections)
            ari_values.append(adjusted_rand_score(y_pred, kmeans.labels_))

        ari_values = np.array(ari_values)

        logger.info("JOB:%d-Plotting ARI" % fold)
        results = pd.DataFrame(
            {
                "Number of clusters (K)": tested_k,
                "Adjusted Rand Index": ari_values,
            }
        )

        # Plotta i risultati con Seaborn
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            data=results,
            x="Number of clusters (K)",
            y="Adjusted Rand Index",
            marker="o",
        )
        plt.title(self.model_name + " ARI")
        plt.tight_layout()
        plt.savefig(self.ckpt_path + "fold=%d_ari.png" % fold)
        plt.close()
        return ari_values

    def explain(
        self,
        save_csv: bool = False,
        plot_pca: bool = False,
        plot_kmeans: bool = False,
        n_jobs: int = 10,
    ):
        results = []

        # Esegui compute_ari per ogni checkpoint in parallelo
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_ari)(int(fold), plot_pca, plot_kmeans)
            for fold in self.checkpoints.keys()
        )

        # Converte i risultati in una matrice numpy
        results = np.array(results)

        df = pd.DataFrame([])

        for fold in self.checkpoints.keys():
            df = df.append(
                pd.DataFrame(
                    {
                        "Number of clusters (K)": list(range(len(results[fold]))),
                        "Adjusted Rand Index": results[fold],
                        "Fold": int(fold),
                    }
                )
            )

        plt.figure(figsize=(8, 6))
        sns.relplot(
            data=df,
            kind="line",
            x="Number of clusters (K)",
            y="Adjusted Rand Index",
            hue="Fold",
        )
        plt.title(self.model_name + " ARI")
        plt.tight_layout()
        plt.savefig(self.ckpt_path + "ari.png")
        plt.close()

        if save_csv:
            df.to_csv(self.ckpt_path + "ari.csv", index=False)

        return df
