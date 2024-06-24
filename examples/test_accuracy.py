import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from pytorch_lightning import seed_everything
from scipy import signal
from tqdm import tqdm
from utils import set_root

seed_everything(42)

import os

# importing
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from physioex.data import Shhs, SleepEDF, TimeDistributedModule, datasets
from physioex.models import load_pretrained_model
from physioex.train.networks import config
from physioex.train.networks import config as networks
from physioex.train.networks.utils.loss import config as losses
from physioex.train.networks.utils.target_transform import get_mid_label

logger.remove()

# model parameters
model_name = "tinysleepnet"
sequence_length = 21

# dataset
picks = ["EEG"]
fold = 0

# dataloader
batch_size = 512
num_workers = 32

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = f"results/{model_name}/"

# load dataset and model
model = networks[model_name]

dataset = Shhs(
    picks=picks,
    sequence_length=sequence_length,
    # target_transform=model["target_transform"],
    target_transform=get_mid_label,
    preprocessing=model["input_transform"],
)

dataset.split(fold=fold)

dataset = TimeDistributedModule(
    dataset=dataset, batch_size=batch_size, fold=fold, num_workers=num_workers
)

model = load_pretrained_model(
    name=model_name,
    in_channels=len(picks),
    sequence_length=sequence_length,
    softmax=True,
).eval()


class MidModel(torch.nn.Module):
    def __init__(self, model):
        super(MidModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, int((x.shape[1] - 1) / 2)]


model = MidModel(model)

# compute the metrics on the test set

dataloder = dataset.test_dataloader()

acc = []
cm = []
reports = []

y_pred = []
pred_probas = []
y_true = []

with torch.no_grad():
    # Aggiungi tqdm per mostrare il progresso
    for i, (inputs, labels) in tqdm(enumerate(dataloder), total=len(dataloder)):
        # Calcola le previsioni del modello

        batch_preds = model(inputs.to(device)).cpu().detach()

        if len(batch_preds.size()) == 2:  # in this case the model is seq to seq
            batch_preds = batch_preds.reshape(-1, batch_preds.shape[-1])
            labels = labels.reshape(-1)

        pred_probas.extend(batch_preds)
        y_pred.extend(torch.argmax(batch_preds, dim=1))
        y_true.extend(labels)


y_true = torch.stack(y_true).numpy()
y_pred = torch.stack(y_pred).numpy()
pred_probas = torch.stack(pred_probas).numpy()

accuracy = accuracy_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Report:")
print(report)

conf_mat_norm = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_mat_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["Wake", "N1", "N2", "N3", "REM"],
    yticklabels=["Wake", "N1", "N2", "N3", "REM"],
)
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# Save all the results in the results folder
fig.savefig(f"{results_dir}confusion_matrix.png")

with open(f"{results_dir}classification_report.txt", "w") as f:
    # write the first line with the accuracy:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    # write the report
    f.write("Report:\n")
    f.write(report)
