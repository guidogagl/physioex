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
from os.path import exists

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

from typing import List

import csv

from captum.attr import IntegratedGradients

def _compute_cross_band_importance(bands : List[List[float]], model : torch.nn.Module, dataloader : D.DataLoader, model_device : torch.device, sampling_rate: int = 100):    

    for i in range(len(bands)):
        assert len(bands[i]) == 2

    y_pred = []
    y_true = []
    band_importance = []
    time_importance = []

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
        filtered_inputs = inputs.reshape(-1, seq_len * n_samples)

        # now inputs size is batch_size * (seq_len * n_channels (1) * n_samples)
        # remove the frequency band from the input using scipy
        
        for band in bands:
            # filter bandstop - reject the frequencies specified in freq_band
            lowcut = band[0]
            highcut = band[1]
            order = 4
            nyq = 0.5 * sampling_rate
            low = lowcut / nyq
            high = highcut / nyq
            sos = signal.butter(order, [low, high], btype='bandstop', output='sos')

            for index in range(batch_size):     
                filtered_inputs[index] = signal.sosfilt(sos, filtered_inputs[index])

        # reshape the input signal to the original size and port it to tensor
        filtered_inputs = filtered_inputs.reshape(batch_size, seq_len, n_channels, n_samples)
        filtered_inputs = torch.from_numpy(filtered_inputs)
        inputs = torch.from_numpy(inputs)

        # compute the prediction of the model with the filtered input, the prediction is a tensor of size batch_size * seq_len, n_classes
        batch_importance = F.softmax(model(filtered_inputs.to(model_device)).cpu()).detach().numpy()

        # the importance is the difference between the prediction with the original input and the prediction with the filtered input
        batch_importance = pred_proba - batch_importance
        band_importance.append(batch_importance)

        ig = IntegratedGradients(model)

        partial_time_importance = []
        for c in range(n_class):
            partial_time_importance.append(ig.attribute(inputs.to(model_device), filtered_inputs.to(model_device), target=c).cpu().numpy())

        partial_time_importance = np.array(partial_time_importance)
        time_importance.append(partial_time_importance)

#    time_importance = np.array(time_importance)

    # reshape the lists to ignore the batch_size dimension
    y_pred = np.concatenate(y_pred).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    band_importance = np.concatenate(band_importance).reshape(-1, n_class)

    #time_importance è una matrice numpy di dimensione batch, n_class, batch_size (32), seq_len (3), n_channels (1), n_samples (3000)
    #faccio un reshape per ottenere una dimensione più comoda da maneggiare

    time_importance = time_importance.reshape((len(dataloader) * batch_size), seq_len, n_samples, n_class)

    return time_importance, band_importance, y_pred, y_true

#RICORDA DI LEVARE I PRIMI DUE PARAMETRI
def compute_band_importance(path, fold, bands : List[List[float]], band_names: List[str],  model : torch.nn.Module, dataloader : D.DataLoader , model_device : torch.device, sampling_rate: int = 100, class_names : list[str] = ["Wake", "N1", "N2", "DS", "REM"], average_type : int = 0):
    
    for i in range(len(bands)):
        assert len(bands[i]) == 2
    assert len(band_names) == 6
    assert len(class_names) == 5
    assert average_type == 0 or average_type == 1

    # compute the cross bands combinations

    dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
    )

    #combinations e' la lista in cui finiranno le varie combinazioni. in particolare e' una lista di liste. ogni elemento della lista e' una
    #lista di combinazioni di bande. il primo elemento e' la lista di combinazioni di 1 banda, il secondo elemento e' la lista di combinazioni di 2 bande, e cosi' via
    band_freq_combinations = []
    band_combinations_dict = {}

    for i in range(len(bands)):
        combination_list = it.combinations(bands, i+1)
        for elem in combination_list:
            band_freq_combinations.append(elem)
 
#    importances_df = []
    #CANCELLARE QUESTO PEZZO DOPO
#    if os.path.exists(path + "band_combinations_importance_fold=" + fold + ".csv"):
#        importances_df = pd.read_csv(path + "band_combinations_importance_fold=" + fold + ".csv")
#        condizione = (importances_df.iloc[:, -6:] == [1, 0, 0, 0, 0, 0]).all(axis=1)
#        y_pred = importances_df.loc[condizione, "y_pred"].values
#        y_true = importances_df.loc[condizione, "y_true"].values
#    else:
    
    for cross_band in band_freq_combinations:
        permuted_bands = np.zeros( len( bands ) )

        for i, band in enumerate( bands ):
            if band in cross_band:
                permuted_bands [i] = 1
        
        print(permuted_bands)
        _, band_importance, y_pred, y_true = _compute_cross_band_importance(cross_band, model, dataloader, model_device, sampling_rate)

        band_combinations_dict[str(permuted_bands)] = band_importance
            #al posto di un dataframe qui mi popolo un dict con le combinazioni delle bande

#            importance_df = pd.DataFrame( band_importance, columns = class_names )

#            importance_df.insert(0, "Sample", range(0, 0 + len(importance_df)))
#            importance_df["y_pred"] = y_pred
#            importance_df["y_true"] = y_true
#            for i, band in enumerate(band_names):
#                importance_df[band] = permuted_bands[i] * np.ones( len(y_pred) )
    
#            importances_df.append( importance_df )
    
#        importances_df = pd.concat( importances_df )

    permuted_bands_importance = []

    # Genera tutte le possibili permutazioni di 0 e 1 lungo 6
    permutations = list(it.product([0, 1], repeat=6))
    # Filtra le permutazioni escludendo quelle con tutti 0
    filtered_permutations = [p for p in permutations if any(x == 1 for x in p)]
    # Converte la lista di tuple in un array NumPy
    permutations_array = np.array(filtered_permutations)

    for i in range(len(permutations_array)):
        key = permutations_array[i]
        permuted_bands_importance.append(band_combinations_dict[str(key)])

    importances_matrix = []

    for i in range(len(bands)):
        #calcolare importanza della banda band per ogni sample

        #simple_average
        if average_type == 0:
            band_importance = get_simple_importance(permuted_bands_importance, permutations_array, i)
        #weighted_average
        elif average_type == 1:
            band_importance = get_weighted_importance(permuted_bands_importance, permutations_array, i)

        importances_matrix.append(band_importance)

    return importances_matrix, y_pred, y_true

def get_simple_importance(permuted_bands_importance : List[np.ndarray], permutations_array : List[List[int]], band : int = 0):
        importance = np.zeros(permuted_bands_importance[0].shape)
        counter = 0

        for i in range(len(permutations_array)):
            if permutations_array[i][band] == 1:
                importance += permuted_bands_importance[i]
                counter += 1

        importance = importance / counter
        return importance

#def get_weighted_importance(permuted_bands_importance : List[np.ndarray], permutations_array : List[List[int]], band : int = 0):
#        importance = np.zeros(permuted_bands_importance[0].shape)

#        for i in range(len(permutations_array)):
#            if permutations_array[i][band] == 1:
#                weight = 1/(np.sum(permutations_array[i] == 1))
#                importance += (permuted_bands_importance[i] * weight)

#        return importance

def get_weighted_importance(permuted_bands_importance : List[np.ndarray], permutations_array : List[List[int]], band : int = 0):
        importance = np.zeros(permuted_bands_importance[0].shape)
        weights_sum = 0

        for i in range(len(permutations_array)):
            if permutations_array[i][band] == 1:
                weight = 1/(np.sum(permutations_array[i] == 1))
                importance += (permuted_bands_importance[i] * weight)
                weights_sum += weight

        importance = importance / weights_sum
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

    def compute_band_time_importance(self, bands : List[List[float]], band_names : List[List[str]], model : torch.nn.Module, dataloader : D.DataLoader, model_device : torch.device, sampling_rate: int = 100, target_band : int = 0, average_type : int = 0, class_names : list[str] = ["Wake", "N1", "N2", "DS", "REM"]):
        
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
        )

        data_iter = iter(dataloader)
        first_batch = next(data_iter)

        band_freq_combinations = []
        target_band_time_cross_importance = []
#        time_combinations_dict = {}
        filtered_permutations = []

        for i in range(len(bands)):
            combination_list = it.combinations(bands, i+1)
            for elem in combination_list:
                band_freq_combinations.append(elem)     

        for cross_band in band_freq_combinations:
            if bands[target_band] not in cross_band:
                continue

            permuted_bands = np.zeros( len( bands ) )

            for i, band in enumerate( bands ):
                if band in cross_band:
                    permuted_bands [i] = 1

            filtered_permutations.append(permuted_bands)

            time_importance, _, y_pred, y_true = _compute_cross_band_importance(cross_band, model, dataloader, model_device, sampling_rate)
        
#            time_combinations_dict[str(permuted_bands)] = time_importance
            target_band_time_cross_importance.append(time_importance)

        permutations_array = np.array(filtered_permutations)
        target_band_time_cross_importance = np.array(target_band_time_cross_importance)

        if average_type == 0:
            target_band_time_importance = get_simple_importance(target_band_time_cross_importance, permutations_array, target_band)
            word = "simple"
        elif average_type == 1:
            target_band_time_importance = get_weighted_importance(target_band_time_cross_importance, permutations_array, target_band)
            word = "weighted"

        #la dimensione di target_band_time_importance è (len(dataloader) * batch_size), seq_len, n_samples, n_class
        #per plottare, provo a plottare solamente la prima matrice di shape seq_len, n_samples, n_class
            
        inputs, y_true_batch = first_batch

        # port the input to numpy
        inputs = inputs.cpu().detach().numpy()
        batch_size, seq_len, n_channels, n_samples = inputs.shape

        inputs = inputs[0].reshape(seq_len, n_samples)
            
        #num_batch, seq_len, n_samples, n_class = target_band_time_importance.shape
                   
        plot_matrix = target_band_time_importance[0]

        for i in range(seq_len):
            y = []
            for j in range(n_samples):
                y.append(plot_matrix[i][j][y_pred[0]])

            x = np.arange(n_samples)

            plt.figure(figsize=(15, 5))
            plt.subplots_adjust(hspace=0.5)

            plt.subplot(2, 1, 1)           
            plt.plot(x, y)
            plt.title("Band " + band_names[target_band] + " time " + word + " importance for the samples of the sequence " + str(i+1) + " of the first batch for predicted class " + class_names[y_pred[0]])

            plt.ylabel('Time Importance')

            x = np.arange(n_samples)
            y = inputs[i]

            plt.subplot(2, 1, 2)
            plt.plot(x, y)
            plt.title("Original corresponding input wave")

            plt.ylabel("Wave value")
            plt.xlabel("Samples")

            plt.savefig(self.ckpt_path + "target_band=" + band_names[target_band] + "_" + word + "_time_importance.png")
            plt.close()

        return target_band_time_importance

    def compute_band_importance(self, bands : List[List[float]], band_names : List[str], fold : int = 0, plot_pred : bool = False, plot_true : bool = False, save_csv : bool = False):
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

        target_band_time_importance = self.compute_band_time_importance(bands, band_names, model, datamodule.train_dataloader(), model_device, self.sampling_rate, 3, 0, self.class_name)

        for i in range(2):
            # RICORDA DI LEVARE I PRIMI DUE PARAMETRI 
            matrixes_importance, y_pred, y_true = compute_band_importance(self.ckpt_path, str(fold), bands, band_names, model, datamodule.train_dataloader(), model_device, self.sampling_rate, self.class_name, i)

            if i == 0:
                word = "simple"
            elif i == 1:
                word = "weighted"

            for j, band in enumerate(band_names):

                if plot_true:
                    ########## plot of simple importance ###########

                    # boxplot of the band simple importance of the true label
                    logger.info("JOB:%d-Plotting band %s %s importance for true label" % (fold, band, word))
                    true_importance = []
                
                    for k in range(len(y_true)):
                        true_importance.append(matrixes_importance[j][k][y_true[k]])
                    
                    true_importance = np.array(true_importance)

                    df = pd.DataFrame({
                        'Band ' + band + ' ' + word + ' Importance': true_importance,
                        'Class': y_true
                    })

                    # boxplot of the true importance of the band with seaborn
                    plt.figure(figsize=(10, 10))
                    ax = sns.boxplot(x='Class', y='Band ' + band + ' ' + word + ' Importance', data=df)
                    ax.set_xticklabels(self.class_name)
                    plt.title('Band ' + band + ' ' + word + ' Importance for True Label')
                    plt.xlabel('Class')
                    plt.ylabel('Importance')
                    plt.savefig(self.ckpt_path + ("fold=%d_true_band=" + band + "_" + word + "_importance.png") % fold)
                    plt.close()

                if plot_pred:
                    ########## plot of simple importance ###########

                    logger.info("JOB:%d-Plotting band %s %s importance for predicted label" % (fold, band, word))
                    pred_importance = []
                    
                    for k in range(len(y_true)):
                        pred_importance.append(matrixes_importance[j][k][y_pred[k]])
                    
                    pred_importance = np.array(pred_importance)

                    df = pd.DataFrame({
                        'Band ' + band + ' ' + word + ' Importance': pred_importance,
                        'Class': y_true
                    })

                    # boxplot of the true importance of the band with seaborn
                    plt.figure(figsize=(10, 10))
                    ax = sns.boxplot(x='Class', y='Band ' + band + ' ' + word + ' Importance', data=df)
                    ax.set_xticklabels(self.class_name)
                    plt.title('Band ' + band + ' ' + word + ' Importance for Predicted Label')
                    plt.xlabel('Class')
                    plt.ylabel('Importance')
                    plt.savefig(self.ckpt_path + ("fold=%d_pred_band=" + band + "_" + word + "_importance.png") % fold)
                    plt.close()

                df_current_average = pd.DataFrame(matrixes_importance[j], columns = self.class_name)
                df_current_average["Predicted Label"] = y_pred
                df_current_average["True Label"] = y_true
                df_current_average["Fold"] = fold

                if save_csv:
                    df_current_average.to_csv(self.ckpt_path + "band=" + band + "_" + word + "_importance.csv", mode = 'a', index=False)
                
        return matrixes_importance
    
    def explain(self, bands : list[list[float]], band_names : list[str], save_csv : bool = False, plot_pred : bool = False, plot_true : bool = False, n_jobs : int = 10):

#        simple_result = []
#        weighted_result = []
#        normalized_result = []

        # Esegui compute_band_importance per ogni checkpoint in parallelo
        result = Parallel(n_jobs=n_jobs)(delayed(self.compute_band_importance)(bands, band_names, int(fold), plot_pred, plot_true, save_csv) for fold in self.checkpoints.keys())

        # Converte i risultati in una matrice numpy
#        simple_result = np.array(simple_result, dtype=object)
#        weighted_result = np.array(weighted_result, dtype=object)
#        normalized_result = np.array(normalized_result, dtype=object)

#       for i in range(len(band_names)):
#           df_simple = pd.DataFrame([])
#           df_weighted = pd.DataFrame([])
#           df_normalized = pd.DataFrame([])

#            for fold in self.checkpoints.keys():
#                df_simple = df_simple.append(pd.DataFrame({
#                   "Band Importance": simple_result[fold][i][:, :-2].tolist(),
#                    "Predicted Label": simple_result[fold][i][:, -2],
#                    "True Label": simple_result[fold][i][:, -1],
#                    "Fold": int(fold)
#                }))

#                df_weighted = df_weighted.append(pd.DataFrame({
#                   "Band Importance": weighted_result[fold][i][:, :-2].tolist(),
#                    "Predicted Label": weighted_result[fold][i][:, -2],
#                    "True Label": weighted_result[fold][i][:, -1],
#                    "Fold": int(fold)
#                }))

#                df_normalized = df_normalized.append(pd.DataFrame({
#                   "Band Importance": normalized_result[fold][i][:, :-2].tolist(),
#                    "Predicted Label": normalized_result[fold][i][:, -2],
#                    "True Label": normalized_result[fold][i][:, -1],
#                    "Fold": int(fold)
#                }))

#            if save_csv:
#                df_simple.to_csv(self.ckpt_path + "band=" + band_names[i] + "_simple_importance.csv", index=False)
#                df_weighted.to_csv(self.ckpt_path + "band=" + band_names[i] + "_weighted_importance.csv", index=False)
#                df_normalized.to_csv(self.ckpt_path + "band=" + band_names[i] + "_normalized_importance.csv", index=False)        
               
        return result