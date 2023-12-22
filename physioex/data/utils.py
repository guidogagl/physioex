import pickle
from loguru import logger
from typing import List
import os


@logger.catch
def read_cache(data_path: str):
    try:
        with open(data_path, "rb") as file:
            logger.info("Reading chache from %s" % (data_path))
            cache = pickle.load(file)
    except FileNotFoundError:
        return {}
    
    return cache

@logger.catch
def write_cache(data_path: str, cache ):
    try:
        with open(data_path, "wb") as file:
            logger.info("Caching dataset into %s" % (data_path))
            pickle.dump(
                cache,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except:
        logger.exception("Exception rised while writing cache file")
    return 