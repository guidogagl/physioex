from physioex.data import preprocess

import importlib

import argparse
from loguru import logger


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

    args = parser.parse_args()

    try:
        module = preprocess[args.dataset]
    except KeyError:
        logger.error(f"Dataset {args.dataset} not found.")

    # run the code in the module

    importlib.import_module(module)
