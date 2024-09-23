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

### CLI

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

For instance lets consider how we extended the Preprocesor class to preprocess the HMC dataset
```python
from physioex.preprocess.preprocessor import Preprocessor

class HMCPreprocessor(Preprocessor):
    def __init__(self, 
            preprocessors_name: List[str] = ["xsleepnet"],
            preprocessors = [xsleepnet_preprocessing],
            preprocessor_shape = [[4, 29, 129]],
            data_folder: str = None
            ):

        # calls the Preprocessor constructor, required at the end of your custom setup
        super().__init__(
            dataset_name="hmc",     # this is the name of the dataset PhysioEx will use 
                                    # as PhysioExDataset( dataset=[dataset_name] )
            signal_shape=[4, 3000], # PhysioEx reads sleep epochs of 30 seconds sampled at 100Hz. 
                                    # 4 Is the total amount of channel available in the dataset
            preprocessors_name=preprocessors_name,
            preprocessors=preprocessors,
            preprocessors_shape=preprocessor_shape,
            data_folder=data_folder,
        )

    @logger.catch
    def download_dataset(self) -> None: 
        # download the dataset into the data_folder/download/hmc_dataset.zip
        # extract the zip 

        download_dir = os.path.join(self.dataset_folder, "download")

        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

            zip_file = os.path.join(self.dataset_folder, "hmc_dataset.zip")

            if not os.path.exists(zip_file):
                download_file(
                    "https://physionet.org/static/published-projects/hmc-sleep-staging/haaglanden-medisch-centrum-sleep-staging-database-1.1.zip",
                    zip_file,
                )

            extract_large_zip(zip_file, download_dir)

    @logger.catch
    def get_subjects_records(self) -> List[str]:
        # read the RECORDS file into the extracted directory and returns the list of the available records
        subjects_dir = os.path.join(
            self.dataset_folder,
            "download",
            "haaglanden-medisch-centrum-sleep-staging-database-1.1",
        )

        records_file = os.path.join(subjects_dir, "RECORDS")

        with open(records_file, "r") as file:
            records = file.readlines()

        records = [record.rstrip("\n") for record in records]

        return records

    @logger.catch
    def read_subject_record(self, record: str) -> Tuple[np.array, np.array]:
        # read each RECORD ( which is an .edf file ) and return its signal and labels
        return read_edf(
            os.path.join(
                self.dataset_folder,
                "download",
                "haaglanden-medisch-centrum-sleep-staging-database-1.1",
                record,
            )
        )

# an example of a read_edf method

def read_edf(file_path):

    stages_path = file_path[:-4] + "_sleepscoring.edf"
    f = pyedflib.EdfReader(stages_path)
    _, _, annt = f.readAnnotations()
    f._close()

    annotations = []
    for a in annt:
        if a in stages_map:
            annotations.append(stages_map.index(a))

    labels = np.reshape(np.array(annotations).astype(int), (-1))

    num_windows = labels.shape[0]

    f = pyedflib.EdfReader(file_path)
    buffer = []

    for indx, modality in enumerate(["EEG C3-M2", "EOG", "EMG chin", "ECG"]):
        if modality == "EOG":

            left = f.getSignalLabels().index("EOG E1-M2")
            right = f.getSignalLabels().index("EOG E2-M2")

            signal = (f.readSignal(left) + f.readSignal(right)).reshape(-1, fs)

        else:

            i = f.getSignalLabels().index(modality)
            fs = int(f.getSampleFrequency(i))
            signal = f.readSignal(i).reshape(-1, fs)

        signal = signal[: num_windows * 30]
        signal = signal.reshape(-1, 30 * fs)

        # NOTE: THE USER IS IN CHARGE TO DO THE RESAMPLING AND BANDPASS filtering on its datasets!
        # resample the signal at 100Hz
        signal = resample(signal, num=30 * 100, axis=1)
        # pass band the signal between 0.3 and 40 Hz
        signal = bandpass_filter(signal, 0.3, 40, 100)

        buffer.append(signal)
    f._close()

    buffer = np.array(buffer)
    buffer = np.transpose(buffer, (1, 0, 2))

    return buffer, labels

```


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
      