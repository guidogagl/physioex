from physioex.train.bin.test import test_script  as test
from physioex.train.bin.train import train_script as train
from physioex.train.bin.finetune import finetune_script as finetune

from torch import set_float32_matmul_precision
set_float32_matmul_precision("medium")
 
from lightning.pytorch import seed_everything
seed_everything(42, workers=True)