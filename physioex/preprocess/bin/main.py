import argparse

from physioex.preprocess.dcsm import DCSMPreprocessor
from physioex.preprocess.hmc import HMCPreprocessor
from physioex.preprocess.mass import MASSPreprocessor
from physioex.preprocess.mesa import MESAPreprocessor
from physioex.preprocess.mros import MROSPreprocessor
from physioex.preprocess.shhs import SHHSPreprocessor
from physioex.preprocess.kornum import KornumPreprocessor
from physioex.preprocess.sleepedf import SLEEPEDFPreprocessor

preprocessors = {
    "dcsm": DCSMPreprocessor,
    "mass": MASSPreprocessor,
    "shhs": SHHSPreprocessor,
    "hmc": HMCPreprocessor,
    "mesa": MESAPreprocessor,
    "mros": MROSPreprocessor,
    "kornum": KornumPreprocessor,
    "sleepedf": SLEEPEDFPreprocessor,
}

import importlib


def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="hmc",
        required=True,
        help="The name of the dataset to preprocess. Expected type: str. Required. Default: 'hmc'",
    )

    parser.add_argument(
        "--data_folder",
        "-df",
        type=str,
        default=None,
        required=False,
        help="The absolute path of the directory where the physioex dataset are stored, if None the home directory is used. Expected type: str. Optional. Default: None",
    )

    parser.add_argument(
        "--preprocessor",
        "-p",
        type=str,
        default=None,
        required=False,
        help="The name of the preprocessor in case of a custom Preprocessor. Needs to extend physioex.preprocess.proprocessor:Preprocessor. Must be passed as a string in the format path.to.preprocessor.module:PreprocessorClass. Expected type: str. Optional. Default: None",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        required=False,
        help="Specify the path to the configuration .yaml file where to store the options to preprocess the dataset with. Expected type: str. Optional. Default: None",
    )

    args = parser.parse_args()

    # convert args to dict
    args = vars(args)

    # a config file is passed update args with the config file
    if args["config"] is not None:
        import yaml

        with open(args["config"], "r") as f:
            config = yaml.safe_load(f)
        args.update(config)

    # if a custom preprocessor is passed import the class
    if args["preprocessor"] is not None:
        module_path, class_name = args["preprocessor"].split(":")
        module = importlib.import_module(module_path)
        preprocessor = getattr(module, class_name)
    else:
        preprocessor = preprocessors[args["dataset"]]

    preprocessor_args = {}
    preprocessor_args["data_folder"] = args["data_folder"]

    if "preprocessor_kwargs" not in args.keys():
        args["preprocessor_kwargs"] = None 

    if args["preprocessor_kwargs"] is not None:

        # if the user specifies preprocessors in the kwargs they need to be imported

        if "preprocessors" in args["preprocessor_kwargs"]:
            custom_preprocessors = args["preprocessor_kwargs"]["preprocessors"]
            for i, preprocessor in enumerate(custom_preprocessors):
                module_path, class_name = preprocessor.split(":")
                module = importlib.import_module(module_path)
                args["preprocessor_kwargs"]["preprocessors"][i] = getattr(
                    module, class_name
                )

        preprocessor_args.update(args["preprocessor_kwargs"])

    preprocessor(**preprocessor_args).run()
