# Preprocess Module

The preprocess module implements a standard API to import custom or benchmark sleep staging datasets into PhysioEx and let them serializable into a `PhysioExDataset`. This functionality is provided by the `physioex.preprocess.preprocessor:Preprocessor` class.

The `Preprocessor` class is designed to facilitate the preprocessing of physiological datasets. This class provides methods for downloading datasets, reading subject records, applying preprocessing functions, and organizing the data into a structured format suitable for machine learning tasks.

### Example Usage

If you want to preprocess a dataset using a preprocessing function `preprocess_fn`

```python
# Preprocessor is an abstract class, you need an implementation of it to use it. 
# This can be you own defined Preprocessor or one of the available into PhysioEx
from physioex.preprocess.hmc import HMCPreprocessor  

# Define preprocessing functions, it's just a Callable method on each signal
from physioex.preprocess.utils.signal import  xsleepnet_preprocessing

# Initialize Preprocessor
preprocessor = HMCPreprocessor(
    preprocessors_name = ["xsleepnet"], # the name of the preprocessor
    preprocessors = [xsleepnet_preprocessing], # the callable preprocessing method
    preprocessor_shape = [[4, 29, 129]], # the output of the signal after preprocessing, 
                                         # the first element (4) depends on the number of 
                                         # channels available in your system. In HMC they are 4.
    data_folder = "/your/data/path/"
)

# Run preprocessing
preprocessor.run()

# at this point you can read the dataset from the disk using PhysioExDataset

from physioex.data import PhysioExDataset

data = PhysioExDataset(
    datasets = ["hmc"],
    preprocessing = "xsleepnet", # can be "raw" also because the Preprocessor will always save also the raw data
    selected_channels = ["EEG", "EOG", "EMG", "ECG"], # in case you want to read all the channels available
    data_folder = "/your/data/path/",
)

# you can now access any sequence of epochs in the dataset

signal, label = data[0]

signal.shape # will be [21 (default sequence lenght), 4, 29, 129]
label.shape # will be [21]
```

#### CLI

If you want to use the standard PhysioEx implementation of the preprocessor, which will save the data in the raw format and in the format proposed by [XSleepNet](https://arxiv.org/abs/2007.05492), you can use the CLI tool, once the library is istalled just type:

```bash
preprocess --dataset hmc --data_folder  "/your/data/path/"
```

The list of available dataset is:

- [SHHS (Sleep Heart Health Study)](https://sleepdata.org/datasets/shhs): A multi-center cohort study designed to investigate the cardiovascular consequences of sleep-disordered breathing.
- [MROS (MrOS Sleep Study)](https://sleepdata.org/datasets/mros): A study focusing on the outcomes of sleep disorders in older men.
- [MESA (Multi-Ethnic Study of Atherosclerosis)](https://sleepdata.org/datasets/mesa): A study examining the prevalence, correlates, and progression of subclinical cardiovascular disease.
- [DCSM (Dreem Challenge Sleep Monitoring)](https://physionet.org/content/dreem/1.0.0/): A dataset from the Dreem Challenge for automatic sleep staging.
- [MASS (Montreal Archive of Sleep Studies)](https://massdb.herokuapp.com/en/): A comprehensive collection of polysomnographic sleep recordings.
- [HMC (Home Monitoring of Cardiorespiratory Health)](https://physionet.org/content/hmc-kinematics/1.0.0/): A dataset for the study of cardiorespiratory health using home monitoring devices.

Note that for the HMC and DCSM dataset the library will take care to download the dataset if not available into `/your/data/path/`.

### Extending the Preprocessor Class

To build you own defined preprocessor you should extend the Preprocessor class.
The list of the methods that the user need to reimplement to extend the Preprocessor class is:

::: physioex.preprocess.preprocessor.Preprocessor
    handler: python
    options:
      members:
        - __init__
        - download_dataset
        - get_subjets_records
        - read_subject_record
        - customize_table
        - get_sets
      show_root_heading: true
      show_source: true
