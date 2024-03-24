import itertools

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from loguru import logger

from physioex.explain import FreqBandsExplainer

sleep_bands = [[0.5, 4], [4, 8], [8, 11.5], [11.5, 15.5], [15.5, 30], [30, 49.5]]
sleep_bands_names = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]

models = ["chambon2018"]
datasets = [
    # {"name": "dreem", "version": "dodh"},
    {"name": "sleep_physionet", "version": "2018"},
]
seqlen = 21
cache = True
batch_size = 32
loss = "cel"

for model, dataset in itertools.product(models, datasets):

    logger.info(f"Explaining {model} on {dataset['name']} {dataset['version']}")
    expl = FreqBandsExplainer(
        model_name=model,
        dataset_name=dataset["name"],
        version=dataset["version"],
        use_cache=cache,
        sequence_lenght=seqlen,
        ckp_path=f"models/{loss}/{model}/seqlen={seqlen}/{dataset['name']}/{dataset['version']}/",
        batch_size=32,
    )

    expl.explain(
        sleep_bands,
        sleep_bands_names,
        plot_class=True,
        save=False,
    )
