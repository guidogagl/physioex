import torch
from pytorch_lightning import LightningModule

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np 
import seaborn as sns
import pandas as pd
import re
import os

import itertools as it

from joblib import Parallel, delayed

from physioex.explain.base import PhysioExplainer

from physioex.train.networks import config 
from physioex.train.networks.utils.loss import config as loss_config
from physioex.data import datasets, TimeDistributedModule

from loguru import logger
from tqdm import tqdm

from torch.utils import data as D
from torch.nn import functional as F
torch.set_float32_matmul_precision('medium')

def compute_band_importance(bands : list[list[float]], model : torch.nn.Module, dataloader : D.DataLoader , model_device : torch.device, sampling_rate: int = 100):
    y_pred = []
    y_true = []
    importance = []

    for batch in dataloader:
        inputs, y_true_batch = batch
        
        # store the true label of the input element
        y_true.append(y_true_batch.numpy())

        # compute the prediction of the model
        pred_proba = F.softmax(model(inputs.to(model_device)).cpu()).detach().numpy()       
        y_pred.append( np.argmax( pred_proba, axis = -1) )
        n_class = pred_proba.shape[-1]

        # port the input to numpy
        inputs = inputs.cpu().detach().numpy()
        batch_size, seq_len, n_channels, n_samples = inputs.shape

        # in our experiments n_channels is always 1
        # in our experiments n_samples is always 3000 ( 30 seconds of data sampled at 100 Hz )
        # in our experiments seq_len is always 3 ( 3 consecutive 30 seconds windows )
        # in our experiments batch_size is always 32 ( the batch size is the number of samples used to compute the gradient )
        # in our experiments the number of classes (y_true) is always 5 ( wake, N1, N2, N3, REM ) each element of y_true is an integer in [0, 4]
        # y_true size = batch_size, 1
        
        # reshape the input to consider only the input signal ( 30 seconds of data sampled at 100 Hz )
        inputs = inputs.reshape(-1, seq_len * n_samples)

        # now inputs size is batch_size * (seq_len * n_channels (1) * n_samples)
        # remove the frequency band from the input using scipy
        
        for band in bands:
            # filter bandstop - reject the frequencies specified in freq_band
            lowcut = bands[band][0]
            highcut = bands[band][1]
            order = 4
            nyq = 0.5 * sampling_rate
            low = lowcut / nyq
            high = highcut / nyq
            sos = signal.butter(order, [low, high], btype='bandstop', output='sos')

            for index in range(batch_size):     
                inputs[index] = signal.sosfilt(sos, inputs[index])

        # reshape the input signal to the original size and port it to tensor
        inputs = inputs.reshape(batch_size, seq_len, n_channels, n_samples)
        inputs = torch.from_numpy(inputs)

        # compute the prediction of the model with the filtered input, the prediction is a tensor of size batch_size * seq_len, n_classes
        batch_importance = F.softmax(model(inputs.to(model_device)).cpu()).detach().numpy()

        # the importance is the difference between the prediction with the original input and the prediction with the filtered input
        batch_importance = pred_proba - batch_importance
        importance.append(batch_importance)

    # reshape the lists to ignore the batch_size dimension

    y_pred = np.concatenate(y_pred).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    importance = np.concatenate(importance).reshape(-1, n_class)

    return importance, y_pred, y_true

def get_normalized_weights(lenght : int):
    weights = []
    sum = 0

    for i in range(lenght):
        weights.append(1/(i+1))
        sum += 1/(i+1)

    weights = np.array(weights)
    weights = weights / sum

    return weights

def get_band_importance(band : str, band_dict : dict):
    shape = 0
    counter = 0
    for key, value in band_dict:
        if shape == 0:
            importance = np.zeros(value.shape)
            shape = 1
        if band in key:
            importance += value
            counter += 1
    
    importance = importance / counter

    return importance

class FreqBandsExplainer(PhysioExplainer):
    def __init__(self,
            model_name : str = "chambon2018", 
            dataset_name : str = "sleep_physioex",
            loss_name : str = "cel", 
            ckp_path : str = None,
            version : str = "2018", 
            use_cache : bool = True, 
            sequence_lenght : int = 3, 
            batch_size : int = 32,
            sampling_rate : int = 100,
            class_name : list = ['Wake', 'NREM1', 'NREM2', 'DeepSleep', 'REM']
        ):
        super().__init__(model_name, dataset_name, loss_name, ckp_path, version, use_cache, sequence_lenght, batch_size)
        self.sampling_rate = sampling_rate
        self.class_name = class_name

    def print_combination(self, band_names : list):
        combination = list(it.combinations(band_names))
        print(combination)

    def compute_band_importance(self, bands : list[list[float]], band_names : list[str], fold : int = 0, plot_pred : bool = False, plot_true : bool = False):
        logger.info("JOB:%d-Loading model %s from checkpoint %s" % (fold, str(self.model_call), self.checkpoints[fold]))
        model = self.model_call.load_from_checkpoint(self.checkpoints[fold], module_config = self.module_config).eval()

        model_device = next(model.parameters()).device

        logger.info("JOB:%d-Splitting dataset into train, validation and test sets" % fold)
        self.dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset = self.dataset, 
            sequence_lenght = self.module_config["seq_len"], 
            batch_size = self.batch_size, 
            transform = self.input_transform, 
            target_transform = self.target_transform
        )

        self.module_config["loss_params"]["class_weights"] = datamodule.class_weights()

        #la band_importance e' un dict che ha come chiavi le combinazioni delle bande, e come valori l'importanza per quella combinazione
        band_importance = {}

        #combinations e' la lista in cui finiranno le varie combinazioni. in particolare e' una lista di liste. ogni elemento della lista e' una
        #lista di combinazioni di bande. il primo elemento e' la lista di combinazioni di 1 banda, il secondo elemento e' la lista di combinazioni di 2 bande, e cosi' via
        band_names_combinations = []
        band_freq_combinations = []

        #pesi normalizzati in base al numero di bande da filtrare
        #weights = get_normalized_weights(len(bands))

        for i in range(len(bands)):
            band_names_combinations.append(list(it.combinations(band_names, i+1)))
            band_freq_combinations.append(list(it.combinations(bands, i+1)))

        #ora, per ogni combinazione possibile di bande, mi calcolo l'importanza e la metto in band_importance, associata alla relativa chiave
        for i in range(len(band_freq_combinations)):
            for j in range(len(band_freq_combinations[i])):
                bands_set = list(band_freq_combinations[i][j])
                importance, y_pred, y_true = compute_band_importance(bands_set, model, datamodule.train_dataloader(), model_device, self.sampling_rate)
                #da moltiplicare per weights[i] per media pesata
                band_importance[str(list(band_names_combinations[i][j]))] = importance

        #ora band_importance e' un dizionario come descritto piu' su                
        
        result = []
        for band in band_names:

            #in base alla banda, ora dovro' prendermi l'importanza di quella banda per poterla plottare. per farlo, devo prendere, dal mio dizionario,
            #tutte le importanze in cui la mia banda compare, e poi farne la media
            importance = get_band_importance(str(band), band_importance)

            if plot_true:
                # boxplot of the band importance of the true label
                logger.info("JOB:%d-Plotting band %s importance for true label" % (fold, band))
                true_importance = []
                
                for i in range(len(y_true)):
                    true_importance.append(importance[i][y_true[i]])
                
                true_importance = np.array(true_importance)

                df = pd.DataFrame({
                    'Band ' + band + ' Importance': true_importance,
                    'Class': y_true
                })

                # boxplot of the true importance of the band with seaborn
                plt.figure(figsize=(10, 10))
                ax = sns.boxplot(x='Class', y='Band ' + band + ' Importance', data=df)
                ax.set_xticklabels(self.class_name)
                plt.title('Band ' + band + ' Importance for True Label')
                plt.xlabel('Class')
                plt.ylabel('Importance')
                plt.savefig(self.ckpt_path + ("fold=%d_true_band=" + band + "_importance.png") % fold)
                plt.close()

            if plot_pred:
                logger.info("JOB:%d-Plotting band %s importance for predicted label" % (fold, band))
                pred_importance = []
                
                for i in range(len(y_true)):
                    pred_importance.append(importance[i][y_pred[i]])
                
                pred_importance = np.array(pred_importance)

                df = pd.DataFrame({
                    'Band ' + band + ' Importance': pred_importance,
                    'Class': y_true
                })

                # boxplot of the true importance of the band with seaborn
                plt.figure(figsize=(10, 10))
                ax = sns.boxplot(x='Class', y='Band ' + band + ' Importance', data=df)
                ax.set_xticklabels(self.class_name)
                plt.title('Band ' + band + ' Importance for Predicted Label')
                plt.xlabel('Class')
                plt.ylabel('Importance')
                plt.savefig(self.ckpt_path + ("fold=%d_pred_band=" + band + "_importance.png") % fold)
                plt.close()

                #result prima era una matrice che aveva, per ogni riga, l'importanza di una banda per ogni classe, affiancata da y_pred e y_true
                #adesso lo rendiamo un array di matrici, ogni posizione dell'array corrisponde a una banda
                #l'array viene inizializzato prima del for, come vuoto
                result.append(np.column_stack([importance, y_pred, y_true]))

        return result
    
    def explain(self, bands : list[list[float]], band_names : list[str], save_csv : bool = False, plot_pred : bool = False, plot_true : bool = False, n_jobs : int = 10):
        results = []

        # Esegui compute_band_importance per ogni checkpoint in parallelo
        results = Parallel(n_jobs=n_jobs)(delayed(self.compute_band_importance)(bands, band_names, int(fold), plot_pred, plot_true) for fold in self.checkpoints.keys())

        # Converte i risultati in una matrice numpy
        results = np.array(results, dtype=object)

        for i in range(len(band_names)):
            df = pd.DataFrame([])

            for fold in self.checkpoints.keys():
                df = df.append(pd.DataFrame({
                   "Band Importance": results[i][fold][:, :-2].tolist(),
                    "Predicted Label": results[i][fold][:, -2],
                    "True Label": results[i][fold][:, -1],
                    "Fold": int(fold)
                }))

            if save_csv:
                df.to_csv(self.ckpt_path + "band=" + band_names[i] + "_importance.csv", index=False)
               
        return df
