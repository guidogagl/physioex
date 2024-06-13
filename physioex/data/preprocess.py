import argparse


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
