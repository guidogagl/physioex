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

    def get_sets(self):
        train, valid, test = super().get_sets()

        np.random.seed(1234)
        np.random.shuffle(train)
        self.exp_split = train[:1600]

        train = train[1600:]

        return train, valid, test

    def get_exp_dataloader(self, batch_size=16):
        self.split()
        self.get_sets()
        
        exp_dataloader = DataLoader(
            self,
            batch_size = batch_size,
            sampler = SubsetRandomSampler(self.exp_split),
            num_workers = 32
        )
        
        return exp_dataloader        
        

class MISO(torch.nn.Module):
    def __init__(self, model):
        super(MISO, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:, int((x.shape[1] - 1) / 2)]

def smooth(x, kernel_size=3):
    return torch.nn.functional.avg_pool1d(
        x, kernel_size=kernel_size, stride=int(kernel_size / 2)
    )

def process_explanations(explanations, kernel_size=300):
    explanations = explanations.squeeze()

    batch_size, seq_len, num_samples, n_bands = explanations.size()

    # consider only the first half of the bands + 1 ( the last is gamma and is not relevant for sleep )
    explanations = explanations[..., : int(n_bands / 2) + 1]

    # consider only the mid epoch of the sequence (the one that is more relevant)
    explanations = explanations[:, int((seq_len - 1) / 2)]

    explanations = torch.permute(explanations, [0, 2, 1])

    # smooth the num_samples dimension
    explanations = (
        smooth(explanations) * kernel_size
    )

    explanations = explanations.reshape(batch_size, -1)

    explanations_sign = torch.sign(explanations)

    explanations = torch.pow(10, torch.abs(explanations))

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
            picks = module_config["picks"],
            sequence_length=module_config["seq_len"],
            target_transform= get_mid_label,
        )
        
        self.mean, self.std = dataset.mean, dataset.std
        self.StandardScaler = StandardScaler(self.mean, self.std)
        
        exp_student_model = torch.nn.Sequential(
            self.StandardScaler,
            self.student,
            torch.nn.Softmax(dim=-1),
        )

                
        self.exp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.student_sg = SpectralGradients(
            exp_student_model, n_bands=module_config["n_bands"]
        )
        
        self.explanations_dataloader = dataset.get_exp_dataloader()
        
        ### compute the explanations for the teacher model
        # try to load the teacher explanations from the disk
        try:
            self.teacher_explanations = torch.load(module_config["teacher_explanations"])
            logger.info("Teacher explanations loaded from disk")
        except FileNotFoundError:  
            logger.info("Teacher explanations not found, computing them")
            self.compute_teacher_explanations(module_config)
            torch.save(self.teacher_explanations, module_config["teacher_explanations"])

        self.register_buffer('teacher_expl', torch.cat(self.teacher_explanations))
        
        self.check_explanations = 0
        self.exp_loss = 0
        
        self.explanations_dataloader = dataset.get_exp_dataloader(batch_size=40)
    
    def compute_teacher_explanations(self, module_config):
 
        teacher = load_pretrained_model(
            name=module_config["teacher"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).nn

        teacher.clf.rnn.train()

        teacher = torch.nn.Sequential(
            self.StandardScaler,
            teacher,
            torch.nn.Softmax(dim=-1),
        )


        # TODO: we know that the teacher is MIMO and we need to omologate it to MISO
        # in general this should be configured by the config file
        teacher = MISO(teacher)

        teacher = SpectralGradients(
            teacher.to(self.exp_device), n_bands=module_config["n_bands"]
        )

        logger.info("Computing the explanations for the teacher model")
        self.teacher_explanations = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.explanations_dataloader):

                exp = teacher.attribute(inputs.to(self.exp_device), target=targets.to(self.exp_device)).detach().cpu()
                self.teacher_explanations.extend( process_explanations(exp, self.kernel_size) )
        
        del teacher
        
        return self.teacher_explanations
                
    def training_step(self, batch, batch_idx):
        # Logica di training
        inputs, targets = batch

        outputs = self.nn( self.StandardScaler(inputs))
        
        if self.check_explanations % 300 == 0:
            student_explanations = []
            with torch.no_grad():
                for inputs, targets in tqdm(self.explanations_dataloader, position=0, leave=True):
                    exp = self.student_sg.attribute(inputs.to(self.exp_device), target=targets.to(self.exp_device)).cpu()
                    student_explanations.append( process_explanations(exp, self.kernel_size) )
            
            student_explanations = torch.cat(student_explanations)
            
            print(student_explanations.size(), self.teacher_expl.size())
            
            self.exp_loss = self.mse(student_explanations, self.teacher_expl)
        
        self.check_explanations += 1
        
        self.log("exp_loss", self.exp_loss, prog_bar=True)
        
        return self.exp_loss + self.compute_loss( outputs, targets)

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

