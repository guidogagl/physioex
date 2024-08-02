import os

from pathlib import Path
from loguru import logger

data_folder = os.path.join( str( Path.home() ), "physioex-data" )

def get_data_folder():
    return data_folder


def set_data_folder(new_data_folder: str):
    global data_folder

    if not Path(new_data_folder).exists():
        logger.warning(f"Path {new_data_folder} does not exist. Trying to create it.")
        try:
            Path(new_data_folder).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create the path {new_data_folder}.")
            logger.error(f"Error: {e}")
            exit()

    data_folder = new_data_folder
    logger.info(f"Data folder set to {data_folder}")

    return data_folder
