import os
from pathlib import Path
from typing import List, Union
import time

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule
from physioex.train.bin.parser import PhysioExParser
from physioex.preprocess.utils.signal import OnlineVariance


def subject_density(prototype_predictions: np.ndarray[np.int_],
                    channels: List[str],
                    n_proto: List[int],
                    subjects: np.ndarray[np.int_],
                    results_path: str):
    
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {'subjects': subjects}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    # Create a barplot with value counts of df['subjects']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['subjects'].value_counts().index, y=df['subjects'].value_counts(normalize=True).values, palette="viridis")
    plt.title("Value Counts of Subjects")
    plt.xlabel("Subjects")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density' / 'subject_density'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'subject_counts.png'))
    plt.close()
    
    for c_idx, c in enumerate(channels):
        ratio_table = df.groupby(c)['subjects'].value_counts(normalize=True).unstack().fillna(0)
        fig, axes = plt.subplots(int(np.ceil(np.sqrt(n_proto[c_idx]))), int(np.floor(np.sqrt(n_proto[c_idx]))), figsize=(25, 25), dpi=300)
        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            if i not in ratio_table.index:
                ax.axis('off')
            else:
                ax.pie(ratio_table.loc[i], labels=ratio_table.loc[i].index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25})
            ax.set_title(f'Prototype {i} - Count {(df[c] == i).sum()}', fontsize=25)
        plt.tight_layout()
        save_path = Path(results_path) / 'prototype_density' / 'subject_density'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{c}.png'))
        
        ratio_table.plot(kind='bar', stacked=True)
        plt.savefig(os.path.join(save_path, f'{c}_bars.png'))


def count_transitions(prototype_predictions: np.ndarray[np.int_],
                      transition_matrix: np.ndarray[np.int_]):
    
    sequence_center = prototype_predictions.shape[1] // 2 + 1
    prototype_predictions = prototype_predictions[:, sequence_center-1 : sequence_center+1, 0]
    
    for r in prototype_predictions:
        transition_matrix[r[0], r[1]] += 1
    
    return transition_matrix
        
        
def average_psd(channels: List[str],
                mean_psd: dict,
                results_path: str) -> None:
    '''Plot the average psd for each prototype and channel'''
    
    n_channels = len(channels)
    cmap = plt.get_cmap("tab10", n_channels)

    for c_idx, c in enumerate(channels):
        n_proto_c = len(mean_psd[c])
        fig, axes = plt.subplots(int(np.ceil(np.sqrt(n_proto_c))), int(np.floor(np.sqrt(n_proto_c))), figsize=(5 * int(np.sqrt(n_proto_c)), 4 * int(np.sqrt(n_proto_c))), dpi=300, tight_layout=True)
        fig_linear, axes_linear = plt.subplots(int(np.ceil(np.sqrt(n_proto_c))), int(np.floor(np.sqrt(n_proto_c))), figsize=(5 * int(np.sqrt(n_proto_c)), 4 * int(np.sqrt(n_proto_c))), dpi=300, tight_layout=True)

        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            ax_linear = axes_linear[int(i // axes.shape[1]), int(i % axes.shape[1])]

            if i < n_proto_c:
                s_mean, s_std = mean_psd[c][i].compute()
                        
                ax.set_xticks(np.linspace(0, s_mean.size - 1, 11))
                ax.set_xticklabels(np.arange(0, 51, 5))
                ax.set_xlim(0, s_mean.size - 1)
                ax.plot(s_mean, color=cmap(c_idx))
                ax.fill_between(np.arange(s_mean.size), s_mean - s_std, s_mean + s_std, alpha=0.5, color=cmap(c_idx))
                ax.set_title(f"P{i} "+ c)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD (dB)")
                
                # Convert s_mean and s_std from dB to linear
                s_mean = 10 ** (s_mean / 20)
                s_std = 10 ** (s_std / 20)
                s_mean = s_mean[:s_mean.size//2]
                s_std = s_std[:s_std.size//2]
                ax_linear.set_xticks(np.linspace(0, s_mean.size - 1, 6))
                ax_linear.set_xticklabels(np.arange(0, 26, 5))
                ax_linear.set_xlim(0, s_mean.size - 1)
                ax_linear.plot(s_mean, color=cmap(c_idx))
                ax_linear.fill_between(np.arange(s_mean.size), s_mean - s_std, s_mean + s_std, alpha=0.5, color=cmap(c_idx))
                ax_linear.set_title(f"P{i} "+ c)
                ax_linear.set_xlabel("Frequency (Hz)")
                ax_linear.set_ylabel("Power (linear)")
            else:
                ax.axis('off')
                ax_linear.axis('off')
        
        fig.savefig(os.path.join(results_path, f'{c}_psd_dB.png'))
        fig_linear.savefig(os.path.join(results_path, f'{c}_psd_linear.png'))
    
        
def prototype_density(prototype_predictions: np.ndarray[np.int_],
                      n_proto: List[int],
                      y: np.ndarray[np.int_],
                      channels: List[str],
                      results_path: str) -> None:
    # Plot the ratio of sleep stage per prototype. One plot per channel. 
    df_dict = {'y': y}
    for c_idx, c in enumerate(channels): 
        df_dict[c] = prototype_predictions[:,c_idx]
    df = pd.DataFrame(df_dict)
    
    for c_idx, c in enumerate(channels):
        ratio_table = df.groupby(c)['y'].value_counts(normalize=True).unstack().fillna(0)
        fig, axes = plt.subplots(int(np.ceil(np.sqrt(n_proto[c_idx]))), int(np.floor(np.sqrt(n_proto[c_idx]))), figsize=(25, 25), dpi=300)
        for i in range(axes.size):
            ax = axes[int(i // axes.shape[1]), int(i % axes.shape[1])]
            if i not in ratio_table.index:
                ax.axis('off')
            else:
                ax.pie(ratio_table.loc[i].sort_index(), labels=ratio_table.loc[i].sort_index().index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 25}, colors=sns.color_palette("Set1"))
                ax.set_title(f'Prototype {i}, {((df[c] == i).sum() / len(y) * 100):.2f}%', fontsize=25)
        plt.tight_layout()
        save_path = Path(results_path) / 'prototype_density' / 'specificity'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{c}.png'))
        ratio_table.to_csv(os.path.join(save_path, f'{c}.csv'))
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 7), dpi=300)
    for c_idx, c in enumerate(channels):
        if isinstance(axes, np.ndarray):
            ax = axes[c_idx]
        else:
            ax = axes
        ratio_table = df.groupby('y')[c].value_counts(normalize=True).unstack().fillna(0).T
        ratio_table = ratio_table.reset_index() 
        melted_df = ratio_table.melt(id_vars=c, var_name='Stage', value_name='Sensitivity')
        for proto in range(n_proto[c_idx]):
            if proto not in melted_df[c].unique():
                melted_df = pd.concat([melted_df, pd.DataFrame({c: [proto], 'Stage': [np.nan], 'Sensitivity': [np.nan]})], ignore_index=True)
        sns.barplot(data=melted_df, x=c, y='Sensitivity', hue='Stage', ax=ax, palette=sns.color_palette("Set1"), hue_order=sorted(melted_df['Stage'].dropna().unique()))
        ax.set_title(c)
        ax.set_xlabel('Prototype')
        ax.legend(title='Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density' / 'sensitivity'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.xticks(rotation=90)  # Rotate x-axis tick labels
    plt.savefig(os.path.join(save_path, 'sensitivity.png'))
    melted_df.to_csv(os.path.join(save_path, 'sensitivity.csv'))
        
    # Plot the counts of each prototype per channel
    value_counts = {c: df[c].value_counts() for c in channels}
    
    bar_data = []
    for c_idx, c in enumerate(channels):
        for idx in range(n_proto[c_idx]):
            bar_data.append({'Channel': c, 'Prototype': idx, 'Count': value_counts[c].get(idx, 0)})
    bar_df = pd.DataFrame(bar_data)
    
    plt.figure(figsize=(16, 8))
    sns.barplot(data=bar_df, x='Prototype', y='Count', hue='Channel', dodge=True, palette=sns.color_palette("Set1"))
    plt.title('Value Counts for EEG, EOG, and EMG')
    plt.xticks(rotation=45)
    plt.legend(title='Channel')
    plt.tight_layout()
    save_path = Path(results_path) / 'prototype_density'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'counts.png'))
    plt.close()

            
def visualize(
    datasets: Union[List[str], str, PhysioExDataModule],
    datamodule_kwargs: dict = {},
    model: SleepModule = None,  # if passed model_class, model_config and resume are ignored
    model_class=None,
    model_config: dict = None,
    batch_size: int = 128,
    fold: int = -1,
    hpc: bool = False,
    checkpoint_path: str = None,
    results_path: str = None,
    num_nodes: int = 1,
    aggregate_datasets: bool = False,
) -> pd.DataFrame:

    # seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    datamodule_kwargs["batch_size"] = batch_size
    # datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes
    # datamodule_kwargs["evaluate_on_whole_night"] = True

    ##### DataModule Setup #####
    if isinstance(datasets, PhysioExDataModule):
        datamodule = [datasets]
    elif isinstance(datasets, str):
        datamodule = [
            PhysioExDataModule(
                datasets=[datasets],
                **datamodule_kwargs,
            )
        ]
    elif isinstance(datasets, list):
        if aggregate_datasets:
            datamodule = PhysioExDataModule(
                datasets=datasets,
                **datamodule_kwargs,
            )
        else:
            datamodule = []
            for dataset in datasets:
                datamodule.append(
                    PhysioExDataModule(
                        datasets=[dataset],
                        **datamodule_kwargs,
                    )
                )
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")

    ########### Resuming Model if needed else instantiate it ############:
    if model is None:
        model = load_model(
            model=model_class,
            model_kwargs=model_config,
            ckpt_path=checkpoint_path,
        )

    ########### Trainer Setup ############
    from lightning.pytorch.accelerators import find_usable_cuda_devices

    devices = find_usable_cuda_devices(-1)

    trainer = Trainer(
        devices=devices,
        strategy="ddp" if (num_nodes > 1 or len(devices) > 1) else "auto",
        num_nodes=num_nodes,
        # callbacks=[progress_bar_callback],
        deterministic=True,
    )
    
    if results_path is not None:
        Path(results_path).mkdir(parents=True, exist_ok=True)
    
    
    learned_prototypes = [chan_codebook.codebook.cpu().detach().numpy() for chan_codebook in model.nn.prototype]
    n_proto = [chan_codebook.shape[0] for chan_codebook in learned_prototypes]
    channels = datamodule_kwargs["selected_channels"]
    n_channels = len(channels)
    assert len(n_proto) == n_channels, "Number of prototypes must match number of channels"
    
    # Create OnlineVariance objects for each prototype and channel
    mean_psd = {}
    transition_matrices = {}
    for c_idx, c in enumerate(channels):
        mean_psd[c] = {}
        for i in range(n_proto[c_idx]):
            mean_psd[c][i] = OnlineVariance((129,))
        
        transition_matrices[c] = np.zeros((n_proto[c_idx], n_proto[c_idx]), dtype=np.int_)
    
    # Initialize a dictionary to store embeddings of each channel
    embeddings = {}
    for c in channels:
        embeddings[c] = []
        
    # Initialize variables to store y, prototype predictions, subject ids
    y = []
    proto_idx = []
    subjects_id = []
    
    for _, test_datamodule in enumerate(datamodule):
        dataloader = test_datamodule.test_dataloader(shuffle=True)
        d_iter = iter(dataloader)
        
        samples = 70000 # number of epochs for visualization
        n_batches = int(samples / batch_size)
        for batch in tqdm(range(n_batches), desc="Getting prototypes"):
            
            # Get the next batch                
            try:
                x_, y_, subject_id_, _ = next(d_iter)
            except StopIteration: # if the iterator is exhausted
                print("End of dataloader after " + str(batch * batch_size) + " samples")
                break
            
            batch_size, L, nchan, T, F = x_.shape
            sequence_center = L // 2 + 1 # index of the center element of the sequence
            
            embeddings_, _, proto_idx_, _, alphas = model.nn.get_prototypes(x_.cuda())
            
            proto_idx_ = proto_idx_.cpu().numpy()
            for c_idx, c in enumerate(channels):
                transition_matrices[c] = count_transitions(proto_idx_[:,:,c_idx], transition_matrices[c])
            
            x_ = x_ * test_datamodule.dataset.readers[0].reader.std + test_datamodule.dataset.readers[0].reader.mean

            embeddings_ = embeddings_[:, sequence_center, :, :, :].detach().cpu().numpy() # take center element of the sequence
            proto_idx_ = proto_idx_[:, sequence_center,:, 0]
            alphas = alphas[:, sequence_center, :, 0, :].detach().cpu().numpy()
            x_ = x_[:, sequence_center].cpu().numpy()
            y_ = y_[:, sequence_center].cpu().numpy()
            
            # select time window that was sampled
            sampled_x = np.einsum('bctf, bct -> bcf', x_, alphas)
                        
            # store psd for each prototype and channel
            for c_idx, c in enumerate(channels):
                for p in np.unique(proto_idx_[:, c_idx]):
                    p_filter_idx = np.where(proto_idx_[:, c_idx] == p)[0]
                    
                    if len(p_filter_idx) > 0:
                        signal = sampled_x[p_filter_idx, c_idx]
                        mean_psd[c][p].add(signal)

            # store embeddings for each channel
            for c_idx, c in enumerate(channels):
                embeddings[c].append(embeddings_[:,c_idx])

            # store y and prototype predictions
            y.append(y_)
            subjects_id.append(subject_id_)
            proto_idx.append(proto_idx_)

        # plot transition matrix
        for c_idx, c in enumerate(channels):
            transition_matrices[c] = transition_matrices[c] / np.sum(transition_matrices[c], axis=1, keepdims=True)
            transition_matrices[c] = np.nan_to_num(transition_matrices[c], nan=0)
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels*max(n_proto)*0.6, max(n_proto)*0.6), dpi=300)
        
        for c_idx, c in enumerate(channels):
            if isinstance(axes, np.ndarray):
                ax = axes[c_idx]
            else:
                ax = axes
            sns.heatmap(transition_matrices[c], annot=True, fmt=".2f", cmap="Blues", ax=ax)
            ax.set_xlabel("Prototype n")
            ax.set_ylabel("Prototype n+1")
            ax.set_title(f"Transition Matrix for {c}")
        plt.tight_layout()
        save_path = Path(results_path) / 'transition_matrix'   
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
        
        # concatenate all batches
        for c_idx, c in enumerate(channels):
            embeddings[c] = np.concatenate(embeddings[c], axis=0)
        y = np.concatenate(y, axis=0)
        subjects_id = np.concatenate(subjects_id, axis=0)
        proto_idx = np.concatenate(proto_idx, axis=0)
                    
        if len(np.unique(y)) == 5:
            label_map = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
            y = np.vectorize(label_map.get)(y)
        elif len(np.unique(y)) == 3:
            label_map = {0: "W", 1: "N", 2: "R"}
            y = np.vectorize(label_map.get)(y)
        
        # plot stage ratio in each prototype and prototype counts
        prototype_density(proto_idx, n_proto, y, channels, results_path)
        
        # plot subject density in each prototype
        subject_density(proto_idx, channels, n_proto, subjects_id, results_path)

        # plot average psd per prototype
        average_psd(channels, mean_psd, results_path)

        # concatenate embeddings and learned prototypes
        for c_idx, c in enumerate(channels):
            embeddings[c] = np.concatenate((np.squeeze(embeddings[c]), learned_prototypes[c_idx]), axis=0)
        
        # plot embeddings and prototypes
        projectors = {'UMAP': UMAP(n_components=2, random_state=42),
                      'PCA': PCA(n_components=2),
                      't-SNE': TSNE(n_components=2, random_state=42)
        }
        
        for k, v in projectors.items():
            print('Computing ' + k)
            initial_time = time.time()
            fig, axes = plt.subplots(2, n_channels, figsize=(10*n_channels, 15))
            if axes.ndim == 1:
                axes = np.expand_dims(axes, axis=1)
            for c_idx, c in enumerate(channels):
                embeddings_2d = v.fit_transform(embeddings[c])
                sns.scatterplot(ax=axes[0, c_idx], x=embeddings_2d[:-n_proto[c_idx], 0], y=embeddings_2d[:-n_proto[c_idx], 1], hue=y, palette=sns.color_palette("bright", len(np.unique(y))))
                
                for p in range(n_proto[c_idx]):
                    axes[0, c_idx].text(embeddings_2d[-n_proto[c_idx]+p, 0], embeddings_2d[-n_proto[c_idx]+p, 1], str(p), fontsize=12, weight='bold', color='black', ha='center', va='center')
                axes[0, c_idx].set_xlabel(k + " Component 1")
                axes[0, c_idx].set_ylabel(k + " Component 2")
                axes[0, c_idx].legend(title="Stage")
                axes[0, c_idx].set_title(c)
                xlim = axes[0, c_idx].get_xlim() # Save current limits because voronoi_plot_2d changes them
                ylim = axes[0, c_idx].get_ylim()
                # vor = Voronoi(embeddings_2d[-n_proto:])
                # voronoi_plot_2d(vor, ax=axes[0, c_idx], show_vertices=False, line_colors='black', line_width=1, show_points=False)
                axes[0, c_idx].set_xlim((xlim[0], xlim[1]*1.25)) # Restore the original limits and make space for legend
                axes[0, c_idx].set_ylim(ylim)
                
                sns.scatterplot(ax=axes[1, c_idx], x=embeddings_2d[:-n_proto[c_idx], 0], y=embeddings_2d[:-n_proto[c_idx], 1], hue=proto_idx[:,c_idx], palette=sns.color_palette("bright", n_proto[c_idx]))
                for p in range(n_proto[c_idx]):
                    axes[1, c_idx].text(embeddings_2d[-n_proto[c_idx]+p, 0], embeddings_2d[-n_proto[c_idx]+p, 1], str(p), fontsize=12, weight='bold', color='black', ha='center', va='center')
                axes[1, c_idx].set_xlabel(k + " Component 1")
                axes[1, c_idx].set_ylabel(k + " Component 2")
                axes[1, c_idx].legend(title="Prototype", ncol=2)
                axes[1, c_idx].set_title(c)
                xlim = axes[1, c_idx].get_xlim() # Save current limits because voronoi_plot_2d changes them
                ylim = axes[1, c_idx].get_ylim()
                # voronoi_plot_2d(vor, ax=axes[1, c_idx], show_vertices=False, line_colors='black', line_width=1, show_points=False)
                axes[1, c_idx].set_xlim((xlim[0], xlim[1]*1.25)) # Restore the original limits and make space for legend
                axes[1, c_idx].set_ylim(ylim)
                
            fig.suptitle(k + " of Embeddings")
            save_path = Path(results_path) / 'embeddings' / k.lower()
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f'{test_datamodule.datasets_id[0]}.png')
            print(f'{k} took {time.time() - initial_time} seconds')




if __name__ == "__main__":
    
    parser = PhysioExParser.test_parser()

    datamodule_kwargs = {
        "selected_channels": parser["selected_channels"],
        "sequence_length": parser["sequence_length"],
        "target_transform": parser["target_transform"],
        "preprocessing": parser["preprocessing"],
        "task": parser["model_task"],
        "data_folder": parser["data_folder"],
        "num_workers": parser["num_workers"],
    }

    visualize(
        datasets=parser["datasets"],
        datamodule_kwargs=datamodule_kwargs,
        model=None,
        fold=parser["fold"],
        model_class=parser["model"],
        model_config=parser["model_kwargs"],
        batch_size=parser["batch_size"],
        hpc=parser["hpc"],
        num_nodes=parser["num_nodes"],
        checkpoint_path=parser["checkpoint_path"],
        results_path=parser["results_path"],
        aggregate_datasets=parser["aggregate"],
    )