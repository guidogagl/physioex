import numpy as np
import torch

from physioex.explain.spectralgradients import SpectralGradients
from physioex.models import load_pretrained_model
from physioex.train.networks.base import SleepModule


class Unstandardize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Unstandardize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x * self.std + self.mean


class Standardize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Standardize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class MISO(torch.nn.Module):
    def __init__(self, model):
        super(MISO, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, int((x.shape[1] - 1) / 2)]


class TeacherStudent(SleepModule):
    def __init__(self, module_config):
        super(TeacherStudent, self).__init__(None, module_config)

        teacher = load_pretrained_model(
            name=module_config["teacher"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).nn

        self.student = load_pretrained_model(
            name=module_config["student"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).nn

        teacher.clf.rnn.train()

        # TODO: we know that the teacher is MIMO and we need to omologate it to MISO
        # in general this should be configured by the config file
        teacher = MISO(teacher)

        # to apply spectral gradients the data need to be unstandardized
        # hence we need to store the mean and std of the data

        # TODO: multi-channel support
        scaling_file = np.load(module_config["scaling_file"])

        self.mean, _, _ = scaling_file["mean"]
        self.std, _, _ = scaling_file["std"]

        self.mean = torch.nn.Parameter(
            torch.tensor(self.mean).float(), requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.tensor(self.std).float(), requires_grad=False
        )

        self.preprocess = Unstandardize(self.mean, self.std)
        self.nn = self.student

        self.mse = torch.nn.MSELoss()
        self.cel = torch.nn.CrossEntropyLoss()

        self.smooth = torch.nn.AvgPool1d(
            module_config["smooth_kernel"],
            stride=int(module_config["smooth_kernel"] / 2),
        )

        # the explanations models are the same but we need to add a softmax activation at the end and a standardization layer at the beginning

        exp_student_model = torch.nn.Sequential(
            Standardize(self.mean, self.std),
            self.student,
            torch.nn.Softmax(dim=-1),
        )

        exp_teacher_model = torch.nn.Sequential(
            Standardize(self.mean, self.std),
            teacher,
            torch.nn.Softmax(dim=-1),
        )

        self.student_sg = SpectralGradients(
            exp_student_model, n_bands=module_config["n_bands"]
        )
        self.teacher_sg = SpectralGradients(
            exp_teacher_model, n_bands=module_config["n_bands"]
        )

    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        outputs = self.nn(inputs)

        # compute the explanations for both models for the target class
        # unstardadize the data
        inputs = self.preprocess(inputs)

        with torch.no_grad():
            explanations_teacher = self.teacher_sg.attribute(inputs, target=targets)
            explanations_student = self.student_sg.attribute(inputs, target=targets)

        return self.compute_loss(
            explanations_student, explanations_teacher, outputs, targets
        )

    def validation_step(self, batch, batch_idx):
        # Logica di validazione
        inputs, targets = batch
        outputs = self.nn(inputs)

        return self.compute_loss(None, None, outputs, targets, "val")

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.nn(inputs)
        return self.compute_loss(None, None, outputs, targets, "test", log_metrics=True)

    def compute_loss(
        self,
        explanations_teacher,
        explanations_student,
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

        if log == "val" or log == "test":
            return cel

        explanations_student, explanations_teacher = (
            explanations_student.squeeze(),
            explanations_teacher.squeeze(),
        )

        batch_size, seq_len, num_samples, n_bands = explanations_teacher.size()

        # consider only the first half of the bands + 1 ( the last is gamma and is not relevant for sleep )
        explanations_teacher = explanations_teacher[..., : int(n_bands / 2) + 1]
        explanations_student = explanations_student[..., : int(n_bands / 2) + 1]

        # consider only the mid epoch of the sequence (the one that is more relevant)
        explanations_teacher = explanations_teacher[:, int((seq_len - 1) / 2)]
        explanations_student = explanations_student[:, int((seq_len - 1) / 2)]

        explanations_teacher = torch.permute(explanations_teacher, [0, 2, 1])
        explanations_student = torch.permute(explanations_student, [0, 2, 1])

        # smooth the num_samples dimension

        explanations_student = (
            self.smooth(explanations_student) * self.smooth.kernel_size[0]
        )
        explanations_teacher = (
            self.smooth(explanations_teacher) * self.smooth.kernel_size[0]
        )

        explanations_student = explanations_student.reshape(batch_size, -1)
        explanations_teacher = explanations_teacher.reshape(batch_size, -1)

        explanations_teacher_sign = torch.sign(explanations_teacher)
        explanations_student_sign = torch.sign(explanations_student)

        explanations_teacher = torch.pow(10, torch.abs(explanations_teacher))
        explanations_student = torch.pow(10, torch.abs(explanations_student))

        # Restore the original sign of the explanations
        explanations_teacher *= explanations_teacher_sign
        explanations_student *= explanations_student_sign

        mse = self.mse(explanations_teacher, explanations_student)

        self.log(f"{log}_mse", mse, prog_bar=True)
        return mse + cel
