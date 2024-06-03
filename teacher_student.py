import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from physioex.data.shhs.shhs import Shhs
from physioex.explain.spectralgradients import SpectralGradients
from physioex.models import load_pretrained_model
from physioex.train.networks.base import SleepModule

from physioex.train.networks.utils.target_transform import get_mid_label


class StandardScaler(torch.nn.Module):
    def __init__(self, mean, std):
        super(StandardScaler, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class TSDataset(Shhs):

    def __getitem__(self, idx):
        x, y = super(Shhs, self).__getitem__(idx)
        y = y - 1

        # return data unstardadized
        # x = (x - self.mean) / self.std
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class MISO(torch.nn.Module):
    def __init__(self, model):
        super(MISO, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, int((x.shape[1] - 1) / 2)]


def smooth(x, kernel_size=3):
    return torch.nn.AvgPool1d(kernel_size=kernel_size, stride=int(kernel_size / 2))(x) 


def process_explanations(explanations, kernel_size=300):
    explanations = explanations.squeeze()

    batch_size, seq_len, num_samples, n_bands = explanations.size()
    # consider only the first half of the bands + 1 ( the last is gamma and is not relevant for sleep )
    explanations = explanations[..., : int(n_bands / 2) + 1]
    # consider only the mid epoch of the sequence (the one that is more relevant)
    explanations = explanations[:, int((seq_len - 1) / 2)]

    explanations = torch.permute(explanations, [0, 2, 1])

    # smooth the num_samples dimension
    explanations = smooth(explanations, kernel_size) * kernel_size

    # check if inf
    if torch.isinf(explanations).any():
        logger.warning("Inf in the explanations")
        
    explanations = explanations.reshape(batch_size, -1)

    explanations_sign = torch.sign(explanations)

    explanations = torch.pow(10, torch.abs(explanations))
    
    # check if inf
    if torch.isinf(explanations).any():
        logger.warning("Inf in the explanations")
        exit()
        
    # Restore the original sign of the explanations
    explanations *= explanations_sign

    return explanations


class TeacherStudent(SleepModule):
    def __init__(self, module_config):
        super(TeacherStudent, self).__init__(None, module_config)

        self.student = load_pretrained_model(
            name=module_config["student"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).nn

        # to apply spectral gradients the data need to be unstandardized
        # hence we need to store the mean and std of the data

        self.nn = self.student

        self.mse = torch.nn.MSELoss()
        self.cel = torch.nn.CrossEntropyLoss()

        self.kernel_size = module_config["smooth_kernel"]

        # the explanations models have softmax at the end
        # and standardize the data at the beginning
        dataset = TSDataset(
            picks=module_config["picks"],
            sequence_length=module_config["seq_len"],
            target_transform=get_mid_label,
        )

        self.mean, self.std = dataset.mean, dataset.std
        self.StandardScaler = StandardScaler(self.mean, self.std)

        student_exp = torch.nn.Sequential(
            self.StandardScaler,
            self.student,
            torch.nn.Softmax(dim=-1),
        )

        self.student_exp = SpectralGradients(
            student_exp, n_bands=module_config["n_bands"]
        )

        teacher_exp = load_pretrained_model(
            name=module_config["teacher"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).nn

        teacher_exp.clf.rnn.train()

        for param in teacher_exp.clf.parameters():
            param.requires_grad = False
        
        teacher_exp = torch.nn.Sequential(
            self.StandardScaler,
            teacher_exp,
            torch.nn.Softmax(dim=-1),
        )

        # TODO: we know that the teacher is MIMO and we need to omologate it to MISO
        # in general this should be configured by the config file
        teacher_exp = MISO(teacher_exp)

        self.teacher_exp = SpectralGradients(
            teacher_exp, n_bands=module_config["n_bands"]
        )


    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        outputs = self.nn(self.StandardScaler(inputs))
        
        with torch.no_grad():
            teacher_explanations = (
                process_explanations(
                    self.teacher_exp.attribute(
                        inputs, target=targets, n_steps=5
                    )
                    .detach()
                    .cpu(),
                    self.kernel_size,
                )
            )

            student_explanations = (
                process_explanations(
                    self.student_exp.attribute(
                        inputs, target=targets, n_steps=5
                    )
                    .detach()
                    .cpu(),
                    self.kernel_size,
                )
            )
        
        self.exp_loss = self.mse(
            student_explanations, teacher_explanations
        )

        self.log("exp_loss", self.exp_loss, prog_bar=True)

        return self.exp_loss + self.compute_loss(outputs, targets)

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self.nn(self.StandardScaler(inputs))

        return self.compute_loss(outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.nn(self.StandardScaler(inputs))
        return self.compute_loss(outputs, targets, "test", log_metrics=True)

    def compute_loss(
        self,
        outputs_student,
        targets,
        log: str = "train",
        log_metrics: bool = False,
    ):
        # print(targets.size())
        batch_size, n_class = outputs_student.size()

        cel = self.cel(outputs_student, targets)

        self.log(f"{log}_cel", cel, prog_bar=True)
        self.log(f"{log}_acc", self.acc(outputs_student, targets), prog_bar=True)

        if log_metrics:
            self.log(f"{log}_f1", self.f1(outputs_student, targets))
            self.log(f"{log}_ck", self.ck(outputs_student, targets))
            self.log(f"{log}_pr", self.pr(outputs_student, targets))
            self.log(f"{log}_rc", self.rc(outputs_student, targets))

        return cel
