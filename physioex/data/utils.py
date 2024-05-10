import os
import pickle

import torch
from loguru import logger

import pkg_resources as pkg

import yaml


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
        return self

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


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
def write_cache(data_path: str, cache):
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


@logger.catch
def read_config(config_path: str):

    config_file = pkg.resource_filename(__name__, config_path)

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config
