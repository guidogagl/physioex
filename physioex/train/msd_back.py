import copy
import json
import uuid
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import sklearn.decomposition as dec
import sklearn.metrics as mt
import torch
from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from physioex.data import MultiSourceDomain as MSD
from physioex.data import TimeDistributedModule
from physioex.data.constant import set_data_folder
from physioex.models import load_pretrained_model
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config


def calculate_combinations(elements):
    combinations_list = []
    for k in range(1, len(elements) + 1):
        combinations_list.extend(combinations(elements, k))
    return combinations_list


torch.set_float32_matmul_precision("medium")

model_dataset = "seqsleepnet"

target_domain = [
    {
        "dataset": "hpap",
        "version": "None",
        "picks": ["EEG"],
    },
    {
        "dataset": "dcsm",
        "version": "None",
        "picks": ["EEG"],
    },
    {
        "dataset": "isruc",
        "version": "None",
        "picks": ["EEG"],
    },
    {
        "dataset": "svuh",
        "version": "None",
        "picks": ["EEG"],
    },
    {
        "dataset": "mass",
        "version": None,
        "picks": ["EEG"],
    },
    {
        "dataset": "dreem",
        "version": "dodh",
        "picks": ["EEG"],
    },
    {
        "dataset": "dreem",
        "version": "dodo",
        "picks": ["EEG"],
    },
    {
        "dataset": "sleep_edf",
        "version": "None",
        "picks": ["EEG"],
    },
    {
        "dataset": "hmc",
        "version": "None",
        "picks": ["EEG"],
    },
]

max_epoch = 10
batch_size = 512
imbalance = False

val_check_interval = 300
num_folds = 1


class MultiSourceDomain:
    def __init__(
        self,
        data_path: str = None,
        model_dataset: str = model_dataset,
        msd_domain: List[Dict] = target_domain,
        sequence_length: int = 21,
        max_epoch: int = max_epoch,
        batch_size: int = batch_size,
        val_check_interval: int = val_check_interval,
        imbalance: bool = imbalance,
        num_folds: int = num_folds,
    ):
        if data_path is not None:
            set_data_folder(data_path)
        seed_everything(42, workers=True)

        self.msd_domain = msd_domain

        self.model_call = config[model_dataset]["module"]

        self.input_transform = config[model_dataset]["input_transform"]
        self.target_transform = config[model_dataset]["target_transform"]
        self.sequence_length = sequence_length
        self.num_folds = num_folds
        self.module_config = config[model_dataset]["module_config"]
        self.module_config["seq_len"] = sequence_length
        self.module_config["in_channels"] = len(msd_domain[0]["picks"])

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval

        self.imbalance = imbalance

        self.module_config["loss_call"] = loss_config["cel"]
        self.module_config["loss_params"] = dict()
        self.k_list = None

    def train_evaluate(self, train_dataset, test_dataset, fold, ckp_path, my_logger):

        logger.info("Splitting datasets into train, validation and test sets")
        train_dataset.split(fold)

        if test_dataset is not None:

            test_dataset.split(fold)
            test_datamodule = TimeDistributedModule(
                dataset=test_dataset,
                batch_size=self.batch_size,
                fold=fold,
            )
        else:
            test_datamodule = None

        datamodule = TimeDistributedModule(
            dataset=train_dataset,
            batch_size=self.batch_size,
            fold=fold,
        )

        progress_bar_callback = RichProgressBar()

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
        )

        logger.info("Trainer setup")
        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True,
            logger=my_logger,
        )

        # check if the model is already on the disk fitted .ckpt file with fold=fold
        # if it is, load it and skip training
        logger.info("Loading the model...")
        list_of_files = list(Path(ckp_path).rglob(f"fold={fold}-*.ckpt"))
        module = load_pretrained_model(name="seqsleepnet").train()

        if len(list_of_files) > 0:
            logger.info("Model already trained, loading model")
            model_path = list_of_files[0]
            module = type(module).load_from_checkpoint(
                model_path, module_config=self.module_config
            )
        else:
            logger.info("Training model")
            trainer.fit(module, datamodule=datamodule)

            # load the best model from the checkpoint callback
            # module = type(module).load_from_checkpoint(checkpoint_callback.best_model_path, module_config=self.module_config).eval()

        logger.info("Evaluating model on train domain")
        train_results = trainer.test(module, datamodule=datamodule)[0]
        train_results["fold"] = fold

        logger.info("Evaluating model on target domain")
        if test_datamodule is not None:
            target_results = trainer.test(module, datamodule=test_datamodule)[0]
            target_results["fold"] = fold

            return {"train_results": train_results, "test_results": target_results}
        return {"train_results": train_results, "test_results": train_results}

    def run_all(self, k_list, n_jobs=1):

        self.k_list = k_list

        domains_id = list(range(len(self.msd_domain)))
        domains_combinations = calculate_combinations(domains_id)

        Parallel(n_jobs=n_jobs)(
            delayed(self.run)(domains_id, combination)
            for combination in domains_combinations
        )

    def run(self, domains_id, combination):

        k = len(combination)

        if k not in self.k_list:  # todo: 5
            return

        train_domain = [self.msd_domain[idx] for idx in combination]
        test_domain = [
            self.msd_domain[idx] for idx in domains_id if idx not in combination
        ]

        train_domain_names = ""
        for domain in train_domain:
            train_domain_names += domain["dataset"]
            train_domain_names += (
                "v." + domain["version"] if domain["version"] is not None else ""
            )

            if domain != train_domain[-1]:
                train_domain_names += "-"

        test_domain_names = ""
        for domain in test_domain:
            test_domain_names += (
                domain["dataset"] + " v. " + domain["version"]
                if domain["version"] is not None
                else "None" + " "
            )
            test_domain_names += "; "

        logger.info("Training on domains: %s" % train_domain_names)
        logger.info("Testing on domains: %s" % test_domain_names)

        train_dataset = MSD(
            domains=train_domain,
            preprocessing=self.input_transform,
            sequence_length=self.sequence_length,
            target_transform=self.target_transform,
            num_folds=self.num_folds,
        )

        if len(test_domain) == 0:
            test_dataset = None
        else:
            test_dataset = MSD(
                domains=test_domain,
                preprocessing=self.input_transform,
                sequence_length=self.sequence_length,
                target_transform=self.target_transform,
                num_folds=self.num_folds,
            )

        results_path = f"models/msd/k={k}/{train_domain_names}/"
        Path(results_path).mkdir(parents=True, exist_ok=True)

        with open(results_path + "domains_setup.txt", "w") as f:
            train_line = "Train domain: " + train_domain_names
            if test_domain is not None:
                test_line = "Test domain: " + test_domain_names
            else:
                test_line = "Test domain: None"

            f.write(train_line + "\n" + test_line)

        results = []
        for fold in range(self.num_folds):

            my_logger = CSVLogger(save_dir=results_path)

            results.append(
                self.train_evaluate(
                    train_dataset, test_dataset, fold, results_path, my_logger
                )
            )
        try:
            train_results = [result["train_results"] for result in results]
            test_results = [result["test_results"] for result in results]
        except:
            return

        pd.DataFrame(train_results).to_csv(
            results_path + "train_results.csv", index=False
        )
        pd.DataFrame(test_results).to_csv(
            results_path + "test_results.csv", index=False
        )

        logger.info("Results successfully saved in %s" % results_path)

    def compute_embeddings_similarity(self, k_list=None):

        domains_id = list(range(len(self.msd_domain)))
        domains_combinations = calculate_combinations(domains_id)

        for combination in domains_combinations:

            k = len(combination)

            if k_list is not None:
                if k not in k_list:
                    continue

            train_domain = [self.msd_domain[idx] for idx in combination]
            test_domain = [
                self.msd_domain[idx] for idx in domains_id if idx not in combination
            ]

            logger.info("Training on domains: %s" % train_domain)

            X_train, y_train, X_test, y_test = get_embeddings(
                train_domain, test_domain, self.sequence_length
            )

            if X_train is None:
                continue

            # random select 128 samples from the train and test embeddings
            idx_train = np.random.choice(X_train.shape[0], 1024, replace=False).astype(
                int
            )
            idx_test = np.random.choice(X_test.shape[0], 1024, replace=False).astype(
                int
            )

            plot_embeddings(train_domain, k, X_train[idx_train], X_test[idx_test])

            logger.info("Computing density estimation...")
            idx_train = np.random.choice(
                X_train.shape[0], 5 * 1024, replace=False
            ).astype(int)
            idx_test = np.random.choice(
                X_test.shape[0], 5 * 1024, replace=False
            ).astype(int)
            X_train, X_test = X_train[idx_train], X_test[idx_test]
            y_train, y_test = y_train[idx_train], y_test[idx_test]

            density = density_estimation(X_train, X_test, y_train, y_test)
            results_df = pd.DataFrame([{"density": density}])

            logger.info(f"Density estimation: {density}")

            train_domain_names = ""
            for domain in train_domain:
                train_domain_names += domain["dataset"]
                train_domain_names += (
                    "v." + domain["version"] if domain["version"] is not None else ""
                )

                if domain != train_domain[-1]:
                    train_domain_names += "-"

            results_path = f"models/msd/k={k}/{train_domain_names}/"

            results_df.to_csv(results_path + "density.csv", index=False)

    def run_age_sex_experiment(
        self,
        task: str = "age",
    ):

        model_call = config["chambon2018"]["module"]
        model_config = config["chambon2018"]["module_config"]

        model_config["seq_len"] = 101
        model_config["in_channels"] = 1

        # customize the model for the task

        if task == "age":
            model_config["n_classes"] = 1
            model_config["loss_call"] = loss_config["reg"]
        elif task == "sex":
            model_config["n_classes"] = 2
            model_config["loss_call"] = loss_config["cel"]
        else:
            logger.error("Task not supported")
            exit()

        model_config["loss_params"] = dict()

        domains = [
            {
                "dataset": "isruc",
                "version": "None",
                "picks": ["EEG"],
            },
            {
                "dataset": "svuh",
                "version": "None",
                "picks": ["EEG"],
            },
            {
                "dataset": "sleep_edf",
                "version": "None",
                "picks": ["EEG"],
            },
        ]

        domains_id = list(range(len(domains)))
        domains_combinations = calculate_combinations(domains_id)

        # insert no combinations at the beginning of the list
        domains_combinations.insert(0, [])

        for combination in domains_combinations:

            k = len(combination)

            if k == 0:
                train_domain = [
                    {"dataset": "hpap", "version": "None", "picks": ["EEG"]}
                ]
            else:
                train_domain = [domains[idx] for idx in combination]

            test_domain = [domains[idx] for idx in domains_id if idx not in combination]

            train_domain_names = ""
            for domain in train_domain:
                train_domain_names += domain["dataset"]
                train_domain_names += (
                    "v." + domain["version"] if domain["version"] is not None else ""
                )

                if domain != train_domain[-1]:
                    train_domain_names += "-"

            test_domain_names = ""
            for domain in test_domain:
                test_domain_names += (
                    domain["dataset"] + " v. " + domain["version"]
                    if domain["version"] is not None
                    else "None" + " "
                )
                test_domain_names += "; "

            logger.info("Training on domains: %s" % train_domain_names)
            logger.info("Testing on domains: %s" % test_domain_names)

            train_dataset = MSD(
                domains=train_domain,
                preprocessing=config["chambon2018"]["input_transform"],
                sequence_length=101,
                target_transform=config["chambon2018"]["target_transform"],
                num_folds=self.num_folds,
                task=task,
            )

            if len(test_domain) != 0:
                test_dataset = MSD(
                    domains=test_domain,
                    preprocessing=config["chambon2018"]["input_transform"],
                    sequence_length=101,
                    target_transform=config["chambon2018"]["target_transform"],
                    num_folds=self.num_folds,
                    task=task,
                )
            else:
                test_dataset = train_dataset

            results_path = f"models/msd_{task}/k={k}/{train_domain_names}/"
            Path(results_path).mkdir(parents=True, exist_ok=True)

            with open(results_path + "domains_setup.txt", "w") as f:
                train_line = "Train domain: " + train_domain_names
                if test_domain is not None:
                    test_line = "Test domain: " + test_domain_names
                else:
                    test_line = "Test domain: None"

                f.write(train_line + "\n" + test_line)

            results = []
            for fold in range(self.num_folds):

                my_logger = CSVLogger(save_dir=results_path)

                logger.info("Splitting datasets into train, validation and test sets")
                train_dataset.split(fold)

                if test_dataset is not None:

                    test_dataset.split(fold)
                    test_datamodule = TimeDistributedModule(
                        dataset=test_dataset,
                        batch_size=256,
                        fold=fold,
                    )
                else:
                    test_datamodule = None

                datamodule = TimeDistributedModule(
                    dataset=train_dataset,
                    batch_size=256,
                    fold=fold,
                )

                progress_bar_callback = RichProgressBar()

                checkpoint_callback = ModelCheckpoint(
                    monitor="val_loss" if task == "age" else "val_acc",
                    save_top_k=1,
                    mode="min" if task == "age" else "max",
                    dirpath=results_path,
                    filename="fold=%d-{epoch}-{step}-{val_loss:.2f}" % fold,
                )

                logger.info("Trainer setup")
                # Configura il trainer con le callback
                trainer = pl.Trainer(
                    max_epochs=self.max_epoch if k != 0 else 50,
                    check_val_every_n_epoch=1,
                    callbacks=[checkpoint_callback, progress_bar_callback],
                    deterministic=True,
                    logger=my_logger,
                )

                # check if the model is already on the disk fitted .ckpt file with fold=fold
                # if it is, load it and skip training
                logger.info("Loading the model...")
                list_of_files = list(Path(results_path).rglob(f"fold={fold}-*.ckpt"))

                if len(list_of_files) > 0:
                    logger.info("Model already trained, loading model")
                    model_path = list_of_files[0]
                    module = model_call.load_from_checkpoint(
                        model_path, module_config=model_config
                    )

                else:
                    logger.info("Training model")
                    try:
                        module = model_call.load_from_checkpoint(
                            "models/msd/k=0/hpapv.None/pretrained.ckpt",
                            module_config=model_config,
                        )
                    except:
                        module = model_call(model_config)

                    trainer.fit(module, datamodule=datamodule)

                logger.info("Evaluating model on train domain")
                train_results = trainer.test(module, datamodule=datamodule)[0]
                train_results["fold"] = fold

                logger.info("Evaluating model on target domain")
                if test_datamodule is not None:
                    target_results = trainer.test(module, datamodule=test_datamodule)[0]
                    target_results["fold"] = fold

                    results = [
                        {"train_results": train_results, "test_results": target_results}
                    ]
                else:
                    results = [
                        {"train_results": train_results, "test_results": train_results}
                    ]

            train_results = [result["train_results"] for result in results]
            test_results = [result["test_results"] for result in results]

            pd.DataFrame(train_results).to_csv(
                results_path + "train_results.csv", index=False
            )
            pd.DataFrame(test_results).to_csv(
                results_path + "test_results.csv", index=False
            )

            logger.info("Results successfully saved in %s" % results_path)


def density_estimation(train_embeddings, test_embeddings, train_labels, test_labels):

    # take the classes which appear in both training and test embeddings
    classes = np.intersect1d(np.unique(train_labels), np.unique(test_labels))

    test_density = 0
    for c in classes:

        X_train = train_embeddings[train_labels == c]
        X_test = test_embeddings[test_labels == c]

        bandwidths = np.linspace(0.01, 0.5, 10)  # Ad esempio, valori da 0.1 a 1.0
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # Configura la ricerca a griglia con validazione incrociata
        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=cv
        )  # Usa 5 fold di validazione incrociata

        # Adatta il modello sui dati
        grid.fit(X_train)
        test_density += grid.score(X_test)

    return test_density / len(test_embeddings)


def plot_embeddings(train_domain, k, train_embeddings, test_embeddings):
    # Supponendo che train_embeddings, test_embeddings, test_labels, e train_labels siano già definiti
    logger.info("Plotting embeddings...")

    train_domain_names = ""
    for domain in train_domain:
        train_domain_names += domain["dataset"]
        train_domain_names += (
            "v." + domain["version"] if domain["version"] is not None else ""
        )

        if domain != train_domain[-1]:
            train_domain_names += "-"

    results_path = f"models/msd/k={k}/{train_domain_names}/"

    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    domains = np.concatenate(
        [np.zeros(train_embeddings.shape[0]), np.ones(test_embeddings.shape[0])], axis=0
    )

    logger.info("Applying PCA...")
    # usa TSNE invece che PCA
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_pca = tsne.fit_transform(embeddings)

    # Creazione di un DataFrame per facilitare il plotting con Seaborn
    df = pd.DataFrame(embeddings_pca, columns=["PCA1", "PCA2"])
    df["Domain"] = domains.astype(
        int
    )  # Convertire i domini in interi per una migliore rappresentazione

    logger.info("Plotting...")
    # Utilizzo di Seaborn per il plotting
    sns.scatterplot(
        data=df, x="PCA1", y="PCA2", hue="Domain", palette="tab10", alpha=0.5
    )

    plt.legend(title="Domain")

    # Se save_path è fornito, salva il plot in quel percorso, altrimenti mostra il plot
    plt.savefig(results_path + "embeddings.png")
    # close the plot
    plt.close()
    return


def encode(model, x, device):
    batch, L, nchan, T, F = x.size()

    x = x.reshape(-1, nchan, T, F)

    x = model.epoch_encoder(x.to(device)).detach().cpu()

    x = x.reshape(batch, L, -1)

    return x


def get_embeddings(train_domain, test_domain, sequence_length):
    k = len(train_domain)

    train_dataset = MSD(
        domains=train_domain,
        preprocessing="xsleepnet",
        sequence_length=sequence_length,
        target_transform=None,
        num_folds=1,
    )

    if len(test_domain) != 0:
        test_dataset = MSD(
            domains=test_domain,
            preprocessing="xsleepnet",
            sequence_length=sequence_length,
            target_transform=None,
            num_folds=1,
        )
    else:
        test_dataset = train_dataset

    train_domain_names = ""
    for domain in train_domain:
        train_domain_names += domain["dataset"]
        train_domain_names += (
            "v." + domain["version"] if domain["version"] is not None else ""
        )

        if domain != train_domain[-1]:
            train_domain_names += "-"

    results_path = f"models/msd/k={k}/{train_domain_names}/"
    # find the pretrained model in the results path
    list_of_files = list(Path(results_path).rglob("*.ckpt"))
    if len(list_of_files) == 0:
        logger.error(f"Model not found in {results_path}")
        return None, None, None, None

    logger.info("Loading the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(name="seqsleepnet")
    model = (
        type(model)
        .load_from_checkpoint(
            list_of_files[0], module_config=config["seqsleepnet"]["module_config"]
        )
        .eval()
    )
    model = model.to(device)

    logger.info("Extracting dataloaders ...")

    train_dataset.split(0)
    test_dataset.split(0)

    train_datamodule = TimeDistributedModule(
        dataset=train_dataset,
        batch_size=512,
        fold=0,
    )

    test_datamodule = TimeDistributedModule(
        dataset=test_dataset,
        batch_size=512,
        fold=0,
    )

    train_dataloader = train_datamodule.train_dataloader()
    test_dataloader = test_datamodule.test_dataloader()

    # consider k * self.batch_size samples to extract the embeddings

    logger.info("Extracting embeddings from train domain...")
    train_embeddings, train_labels = [], []
    k_iter = 2
    for batch in tqdm(train_dataloader, desc="Train domain", total=k_iter):
        X, y = batch
        batch_size = X.size(0)

        embeddings = encode(model.nn, X, device)

        embeddings = embeddings.cpu().detach().numpy()

        embeddings = np.reshape(embeddings, (batch_size * sequence_length, -1))

        train_embeddings.extend(embeddings)
        train_labels.extend(np.reshape(y, -1))

        k_iter -= 1
        if k_iter <= 0:
            break
    train_embeddings = np.array(train_embeddings)
    train_labels = np.array(train_labels)

    logger.info("Extracting embeddings from test domain...")
    k_iter = 2
    test_embeddings, test_labels = [], []
    for batch in tqdm(test_dataloader, desc="Test domain", total=k_iter):
        X, y = batch
        batch_size = X.size(0)

        embeddings = encode(model.nn, X, device)

        embeddings = embeddings.cpu().detach().numpy()

        embeddings = np.reshape(embeddings, (batch_size * sequence_length, -1))

        test_embeddings.extend(embeddings)
        test_labels.extend(np.reshape(y, -1))

        k_iter -= 1
        if k_iter <= 0:
            break

    model = model.to("cpu")

    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)

    return train_embeddings, train_labels, test_embeddings, test_labels


import argparse

if __name__ == "__main__":
    # leggi il task da linea di comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sleep", help="Task to perform")
    parser.add_argument(
        "--data_path", type=str, default="/home/guido/shared/", help="Data path"
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="List of k values to consider for the experiment",
    )

    args = parser.parse_args()

    ssd = MultiSourceDomain(data_path=args.data_path)
    # convert the k_list string into a list of integers
    k_list = [int(k) for k in args.k_list.split(",")]
    logger.info(f"Running the experiment with k values: {k_list}")
    if args.task == "sleep":
        ssd.run_all(k_list)
    elif args.task in ["age", "sex"]:
        ssd.run_age_sex_experiment(task=args.task)
    elif args.task == "density":
        ssd.compute_embeddings_similarity(k_list)
