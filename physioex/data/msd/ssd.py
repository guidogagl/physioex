from physioex.data.base import PhysioExDataset

from physioex.data.shhs.shhs import Shhs
from physioex.data.sleep_edf.sleep_edf import SleepEDF
from physioex.data.dreem.dreem import Dreem

from typing import List, Callable


class SingleSourceDomain(PhysioExDataset):  # SSD
    def __init__(
        self,
        version: str = None,
        picks: List[str] = ["EEG"],  # available [ "EEG", "EOG", "EMG" ]
        preprocessing: str = "xsleepnet",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 21,
        target_transform: Callable = None,
    ):

        assert preprocessing in [
            "raw",
            "xsleepnet",
        ], "preprocessing should be one of 'raw'-'xsleepnet'"

        for pick in picks:
            assert pick in [
                "EEG",
                "EOG",
                "EMG",
            ], "pick should be one of 'EEG, 'EOG', 'EMG'"

        self.train_domain = Shhs(
            picks=picks,
            preprocessing=preprocessing,
            sequence_length=sequence_length,
            target_transform=target_transform,
        )

        self.test_domain = [
            SleepEDF(
                version="2018",
                picks=picks,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                target_transform=target_transform,
            ),
            SleepEDF(
                version="2013",
                picks=picks,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                target_transform=target_transform,
            ),
            Dreem(
                version="dodh",
                picks=picks,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                target_transform=target_transform,
            ),
            Dreem(
                version="dodo",
                picks=picks,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                target_transform=target_transform,
            ),
        ]
