import argparse

from physioex.data.dcsm.preprocess import DCSMPreprocessor
from physioex.data.dreem.preprocess import DREEMPreprocessor
from physioex.data.hmc.preprocess import HMCPreprocessor
from physioex.data.isruc.preprocess import ISRUCPreprocessor
from physioex.data.mass.preprocess import MASSPreprocessor
from physioex.data.shhs.preprocess import SHHSPreprocessor
from physioex.data.sleep_edf.preprocess import SLEEPEDFPreprocessor
from physioex.data.svuh.preprocess import SVUHPreprocessor

preprocessors = {
    "dcsm": DCSMPreprocessor,
    "dreem": DREEMPreprocessor,
    "isruc": ISRUCPreprocessor,
    "mass": MASSPreprocessor,
    "shhs": SHHSPreprocessor,
    "sleep_edf": SLEEPEDFPreprocessor,
    "svuh": SVUHPreprocessor,
    "hmc": HMCPreprocessor,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="sleep_physionet",
        required=True,
        help="The name of the dataset to preprocess. Expected type: str. Required. Default: 'sleep_physionet'",
    )

    parser.add_argument(
        "--data_folder",
        "-df",
        type=str,
        default=None,
        required=False,
        help="The absolute path of the directory where the physioex dataset are stored, if None the home directory is used. Expected type: str. Optional. Default: None",
    )

    args = parser.parse_args()

    preprocessors[args.dataset](data_folder=args.data_folder).run()
