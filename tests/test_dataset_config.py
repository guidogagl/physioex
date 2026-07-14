"""
Unit tests for PhysioExDataset.get_parameters_from_config() method.

Tests the fix for NameError when PHYSIOEX_CONFIG.yaml is missing or malformed.
Run with: python test/tests/test_dataset_config.py
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

from physioex.data.dataset import PhysioExDataset


def make_mock_instance():
    """Create a mock instance that has the attributes get_parameters_from_config expects."""
    instance = MagicMock(spec=PhysioExDataset)
    instance.datasets = ["default_dataset"]
    instance.preprocessing = "raw"
    instance.seqlen = 21
    instance.indexed_channels = ["EEG", "EOG", "EMG", "ECG"]
    instance.selected_channels = ["EEG"]
    instance.channels_index = [0]
    instance.data_folder = "/some/default/path"
    return instance


def test_no_yaml_file():
    """Test 1: No YAML file present -- should not raise NameError."""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # There is no PHYSIOEX_CONFIG.yaml in this temp directory
            instance = make_mock_instance()
            # Call the real unbound method on the mock instance
            PhysioExDataset.get_parameters_from_config(instance)
            # If we get here without NameError, the fix works
            # Attributes should remain unchanged
            assert instance.datasets == ["default_dataset"], f"datasets changed unexpectedly: {instance.datasets}"
            assert instance.seqlen == 21, f"seqlen changed unexpectedly: {instance.seqlen}"
            print("TEST 1 PASS: No YAML file -- no NameError, attributes unchanged.")
    finally:
        os.chdir(original_cwd)


def test_empty_yaml_file():
    """Test 2: YAML file present but empty -- should not crash."""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # Write an empty YAML file
            with open("PHYSIOEX_CONFIG.yaml", "w") as f:
                f.write("")
            instance = make_mock_instance()
            PhysioExDataset.get_parameters_from_config(instance)
            # Attributes should remain unchanged
            assert instance.datasets == ["default_dataset"], f"datasets changed unexpectedly: {instance.datasets}"
            assert instance.seqlen == 21, f"seqlen changed unexpectedly: {instance.seqlen}"
            print("TEST 2 PASS: Empty YAML file -- no crash, attributes unchanged.")
    finally:
        os.chdir(original_cwd)


def test_yaml_with_physioex_section():
    """Test 3: YAML with PhysioExDataset section containing seqlen: 35."""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with open("PHYSIOEX_CONFIG.yaml", "w") as f:
                f.write("PhysioExDataset:\n  seqlen: 35\n")
            instance = make_mock_instance()
            PhysioExDataset.get_parameters_from_config(instance)
            assert instance.seqlen == 35, f"Expected seqlen=35, got {instance.seqlen}"
            # Other attributes should remain unchanged
            assert instance.datasets == ["default_dataset"], f"datasets changed unexpectedly: {instance.datasets}"
            print("TEST 3 PASS: YAML with PhysioExDataset section -- seqlen updated to 35.")
    finally:
        os.chdir(original_cwd)


def test_yaml_missing_physioex_section():
    """Test 4: YAML present but missing PhysioExDataset section -- should not crash."""
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with open("PHYSIOEX_CONFIG.yaml", "w") as f:
                f.write("Trainer:\n  max_epochs: 10\n")
            instance = make_mock_instance()
            PhysioExDataset.get_parameters_from_config(instance)
            # Attributes should remain at defaults
            assert instance.datasets == ["default_dataset"], f"datasets changed unexpectedly: {instance.datasets}"
            assert instance.seqlen == 21, f"seqlen changed unexpectedly: {instance.seqlen}"
            print("TEST 4 PASS: YAML without PhysioExDataset section -- no crash, attributes unchanged.")
    finally:
        os.chdir(original_cwd)


