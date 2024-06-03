import argparse
import importlib
from pathlib import Path

from loguru import logger

from physioex.data.constant import set_data_folder

preprocess = {
    "sleep_physionet": "physioex.data.sleep_edf.preprocess",
    "dreem": "physioex.data.dreem.preprocess",
    "shhs": "physioex.data.shhs.preprocess",
    "mass": "physioex.data.mass.preprocess",
}


def main():
    global data_folder

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

    if args.data_folder is not None:
        # check if the path in args is a valid path
        if not Path(args.data_folder).exists():
            logger.warning(
                f"Path {args.data_folder} does not exist. Trying to create it."
            )
            try:
                Path(args.data_folder).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Could not create the path {args.data_folder}.")
                logger.error(f"Error: {e}")
                return

        set_data_folder(args.data_folder)
        logger.info(f"Data folder set to {args.data_folder}")

    try:
        module = preprocess[args.dataset]
    except KeyError:
        logger.error(f"Dataset {args.dataset} not found.")

    # run the code in the module

    importlib.import_module(module)
