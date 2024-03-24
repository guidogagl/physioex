import gc
import math
import os
import pickle
import shutil
import sys
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, lfilter, resample_poly
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from wfdb import processing


def evaluate_12ECG_score(label_directory, output_directory):
    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = "weights.csv"
    normal_class = "426783006"
    equivalent_classes = [
        ["713427006", "59118001"],
        ["284470004", "63593006"],
        ["427172004", "17338001"],
    ]

    # Find the label and output files.
    print("Finding label and output files...")
    label_files, output_files = find_challenge_files(label_directory, output_directory)

    # Load the labels and outputs.
    print("Loading labels and outputs...")
    label_classes, labels = load_labels(label_files, normal_class, equivalent_classes)
    output_classes, binary_outputs, scalar_outputs = load_outputs(
        output_files, normal_class, equivalent_classes
    )

    # Organize/sort the labels and outputs.
    print("Organizing labels and outputs...")
    classes, labels, binary_outputs, scalar_outputs = organize_labels_outputs(
        label_classes, output_classes, labels, binary_outputs, scalar_outputs
    )

    # Load the weights for the Challenge metric.
    print("Loading weights...")
    weights = load_weights(weights_file, classes)

    # Only consider classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    scalar_outputs = scalar_outputs[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]

    # Evaluate the model by comparing the labels and outputs.
    print("Evaluating model...")

    print("- AUROC and AUPRC...")
    auroc, auprc = compute_auc(labels, scalar_outputs)

    print("- Accuracy...")
    accuracy = compute_accuracy(labels, binary_outputs)

    print("- F-measure...")
    f_measure = compute_f_measure(labels, binary_outputs)

    print("- F-beta and G-beta measures...")
    f_beta_measure, g_beta_measure = compute_beta_measures(
        labels, binary_outputs, beta=2
    )

    print("- Challenge metric...")
    challenge_metric = compute_challenge_metric(
        weights, labels, binary_outputs, classes, normal_class
    )

    print("Done.")

    # Return the results.
    return (
        auroc,
        auprc,
        accuracy,
        f_measure,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    )


# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# Find Challenge files.
def find_challenge_files(label_directory, output_directory):
    label_files = list()
    output_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f)  # Full path for label file
        if (
            os.path.isfile(F)
            and F.lower().endswith(".hea")
            and not f.lower().startswith(".")
        ):
            root, ext = os.path.splitext(f)
            g = root + ".csv"
            G = os.path.join(
                output_directory, g
            )  # Full path for corresponding output file
            if os.path.isfile(G):
                label_files.append(F)
                output_files.append(G)
            else:
                raise IOError(
                    "Output file {} not found for label file {}.".format(g, f)
                )

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError("No label or output files found.")


# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], "r") as f:
            for l in f:
                if l.startswith("#Dx"):
                    dxs = set(arr.strip() for arr in l.split(": ")[1].split(","))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print(
            "- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.".format(
                normal_class
            )
        )
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes) > 1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels[:, representative_index] = np.any(
                labels[:, equivalent_indices], axis=1
            )
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels = np.delete(labels, remove_indices, axis=1)

    return classes, labels


# Load outputs from output files.
def load_outputs(output_files, normal_class, equivalent_classes_collection):
    # The outputs should have the following form:
    #
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(output_files)

    tmp_labels = list()
    tmp_binary_outputs = list()
    tmp_scalar_outputs = list()
    for i in range(num_recordings):
        with open(output_files[i], "r") as f:
            for j, l in enumerate(f):
                arrs = [arr.strip() for arr in l.split(",")]
                if j == 1:
                    row = arrs
                    tmp_labels.append(row)
                elif j == 2:
                    row = list()
                    for arr in arrs:
                        number = 1 if arr in ("1", "True", "true", "T", "t") else 0
                        row.append(number)
                    tmp_binary_outputs.append(row)
                elif j == 3:
                    row = list()
                    for arr in arrs:
                        number = float(arr) if is_number(arr) else 0
                        row.append(number)
                    tmp_scalar_outputs.append(row)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print(
            "- The normal class {} is not one of the output classes, so it has been automatically added, but please check that you identified the correct normal class.".format(
                normal_class
            )
        )
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for k, dx in enumerate(dxs):
            j = classes.index(dx)
            binary_outputs[i, j] = tmp_binary_outputs[i][k]
            scalar_outputs[i, j] = tmp_scalar_outputs[i][k]

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The binary output for the representative class is positive if any of the classes in the set is positive.
    # The scalar output is the mean of the scalar outputs for the classes in the set.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes) > 1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            binary_outputs[:, representative_index] = np.any(
                binary_outputs[:, equivalent_indices], axis=1
            )
            scalar_outputs[:, representative_index] = np.nanmean(
                scalar_outputs[:, equivalent_indices], axis=1
            )
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    binary_outputs = np.delete(binary_outputs, remove_indices, axis=1)
    scalar_outputs = np.delete(scalar_outputs, remove_indices, axis=1)

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    return classes, binary_outputs, scalar_outputs


# Organize labels and outputs.
def organize_labels_outputs(
    label_classes, output_classes, tmp_labels, tmp_binary_outputs, tmp_scalar_outputs
):
    # Include all classes from either the labels or the outputs.
    classes = sorted(set(label_classes) | set(output_classes))
    num_classes = len(classes)

    # Check that the labels and outputs have the same numbers of recordings.
    assert len(tmp_labels) == len(tmp_binary_outputs) == len(tmp_scalar_outputs)
    num_recordings = len(tmp_labels)

    # Rearrange the columns of the labels and the outputs to be consistent with the order of the classes.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for k, dx in enumerate(label_classes):
        j = classes.index(dx)
        labels[:, j] = tmp_labels[:, k]

    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for k, dx in enumerate(output_classes):
        j = classes.index(dx)
        binary_outputs[:, j] = tmp_binary_outputs[:, k]
        scalar_outputs[:, j] = tmp_scalar_outputs[:, k]

    return classes, labels, binary_outputs, scalar_outputs


# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, "r") as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(",")]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception("The table {} is empty.".format(table_file))

    num_cols = set(len(table[i]) - 1 for i in range(num_rows))
    if len(num_cols) != 1:
        raise Exception(
            "The table {} has rows with different lengths.".format(table_file)
        )
    num_cols = min(num_cols)
    if num_cols < 1:
        raise Exception("The table {} is empty.".format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float("nan")

    return rows, cols, values


# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert rows == cols
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights


# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")

    return A


# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure


# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(labels, outputs, beta):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1 + beta**2) * tp + fp + beta**2 * fn:
            f_beta_measure[k] = float((1 + beta**2) * tp) / float(
                (1 + beta**2) * tp + fp + beta**2 * fn
            )
        else:
            f_beta_measure[k] = float("nan")
        if tp + fp + beta * fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
        else:
            g_beta_measure[k] = float("nan")

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc


# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(
            max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1)
        )
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(
            correct_score - inactive_score
        )
    else:
        normalized_score = float("nan")

    return normalized_score


def bandpass_filter(data, lowcut=0.001, highcut=15.0, signal_freq=500, filter_order=1):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def load_challenge_data(filename):

    x = loadmat(filename)
    # print(x)
    data = np.asarray(x["val"], dtype=np.float64)

    new_file = filename.replace(".mat", ".hea")
    input_header_file = os.path.join(new_file)

    with open(input_header_file, "r") as f:
        header_data = f.readlines()

    return data, header_data


# Find unique true labels
def get_true_labels(input_file, classes, classes_cases):

    classes_label = classes
    single_recording_labels = np.zeros(len(classes), dtype=int)
    scored_classes_flag = False
    with open(input_file, "r") as f:
        first_line = f.readline()
        recording_label = first_line.split(" ")[0]
        # print(recording_label)
        for lines in f:
            if lines.startswith("#Dx"):
                tmp = lines.split(": ")[1].split(",")
                for c in tmp:
                    current_class = int(c.strip())

                    if current_class in classes_label:
                        scored_classes_flag = True
                        idx = classes.index(current_class)
                        if classes_cases[idx] > 0:
                            classes_cases[idx] -= 1
                            single_recording_labels[idx] = 1

    return scored_classes_flag, recording_label, classes_label, single_recording_labels


def extend_ts(ts, length):
    extended = np.zeros(length)
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended


def readData(input_files, frame_len, labels, classes, scored_classes):
    fs = 500

    num_classes = len(classes)

    # __________________________________________________________________
    # Data from all mat files.
    # __________________________________________________________________
    num_files = len(input_files)
    multi_labels = []
    X = []
    y = []

    normalize = partial(processing.normalize_bound, lb=-1, ub=1)
    # Iterate over files.
    for i, f in enumerate(input_files):

        # print('    {}/{}...'.format(i+1, num_files))
        # Creating temporary variables for current signal and label
        temp_x = np.zeros((1, frame_len), dtype=np.float32)
        temp_y = np.zeros((1), dtype=int)
        multi_labels_temp = np.zeros((num_classes), dtype=int)
        # Mat files. (ECG data)
        tmp_input_file = f
        data, header_data = load_challenge_data(tmp_input_file)

        # ___________________________________________________________________________
        # Reading Header data and processing it
        #

        # Header files. (ECG Labels)
        g = f.replace(".mat", ".hea")
        tmp_input_file = g

        # Read sampled frequency
        with open(tmp_input_file, "r") as f:
            first_line = f.readline()
            sampled_fs = int(first_line.split(" ")[2])
        # If sample frequency is not 500. Resample data
        if sampled_fs != fs:
            data = resample_poly(data, fs, sampled_fs, axis=1)

        # ___________________________________________________________________________
        # Reading Signal data and processing it
        #

        # If length of ecg signal is greater than the frame length just truncate it.
        if data.shape[1] > frame_len:
            data = data[:, :frame_len]

        extended_data = np.zeros((num_leads, frame_len))

        for j in range(num_leads):

            # If all values in a lead are not zero.
            if data[j, :].any():
                # Frame Normalization
                data[j, :] = np.squeeze(np.apply_along_axis(normalize, 0, data[j, :]))

            # padding zeros and bandpass filtering.
            data[j, :] = bandpass_filter(data[j, :])
            extended_data[j, :] = extend_ts(data[j, :], length=frame_len)

        temp_x = extended_data.T

        # ___________________________________________________________________________
        # Finalizing Labels and Signals into X and y
        #

        # Creating multiple Xs and ys for multi labelled files
        temp_y = labels[i]
        y.append(temp_y)
        X.append(temp_x)
        multi_labels_temp[labels[i]] = 1
        multi_labels.append(multi_labels_temp)

    # Collect unused Variables
    gc.collect()
    del extended_data, input_files, data

    # Converting Python lists to final Training Array
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int)
    multi_labels = np.asarray(multi_labels, dtype=np.int)

    return X, y, multi_labels


def readFiles(input_directory):

    # Getting All folders in Input_directory
    folders = [
        dI
        for dI in os.listdir(input_directory)
        if os.path.isdir(os.path.join(input_directory, dI))
    ]

    # __________________________________________________________________
    # Find all mat files in data directory.
    # __________________________________________________________________
    input_files = []

    for folder in folders:
        current_folder = os.path.join(input_directory, folder)
        for f in os.listdir(current_folder):
            if (
                os.path.isfile(os.path.join(current_folder, f))
                and not f.lower().startswith(".")
                and f.lower().endswith("mat")
            ):
                input_files.append(os.path.join(current_folder, f))

    return input_files


def readFilesWithLabels(input_directory, classes, classes_cases):

    # Getting All folders in Input_directory
    folders = [
        dI
        for dI in os.listdir(input_directory)
        if os.path.isdir(os.path.join(input_directory, dI))
    ]

    # __________________________________________________________________
    # Find all mat files in data directory.
    # __________________________________________________________________
    input_files = []
    labels = []
    for folder in folders:

        current_folder = os.path.join(input_directory, folder)
        for f in os.listdir(current_folder):
            if (
                os.path.isfile(os.path.join(current_folder, f))
                and not f.lower().startswith(".")
                and f.lower().endswith("mat")
            ):

                current_file = os.path.join(current_folder, f)
                # ___________________________________________________________________________
                # Reading Header data and processing it
                #

                # Header files. (ECG Labels)
                g = current_file.replace(".mat", ".hea")
                tmp_input_file = g

                # Check if the current class is in scored classes. Otherwise Skip the file
                (
                    scored_classes_flag,
                    recording_label,
                    classes_label,
                    multi_labels_temp,
                ) = get_true_labels(tmp_input_file, classes, classes_cases)

                # Skipping the files where no scored class label is found
                if not scored_classes_flag:
                    # print("No Scored Class Found in this file")
                    continue

                # Taking All indexes where class label is 1
                idx = np.where(multi_labels_temp == 1)
                # Creating multiple filename entries and labels for multi labelled files
                for i in range(len(idx[0])):
                    temp_y = idx[0][i]
                    input_files.append(current_file)
                    labels.append(temp_y)

    return input_files, labels


def _bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:

        layer = Dropout(params["conv_dropout"])(layer)

    return layer


def add_conv_weight(layer, filter_length, num_filters, subsample_length=1, **params):

    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding="same",
        kernel_initializer=params["conv_init"],
    )(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            params["conv_num_filters_start"],
            subsample_length=subsample_length,
            **params
        )
        layer = _bn_relu(layer, **params)
    return layer


def resnet_block(layer, num_filters, subsample_length, block_index, **params):

    def zeropad(x):
        y = tf.zeros_like(x)
        return tf.concat([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (
        block_index % params["conv_increase_channels_at"]
    ) == 0 and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer, dropout=params["conv_dropout"] if i > 0 else 0, **params
            )
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params
        )
    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index, num_start_filters, **params):
    return 2 ** int(index / params["conv_increase_channels_at"]) * num_start_filters


def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params
    )
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params
        )
        layer = resnet_block(layer, num_filters, subsample_length, index, **params)
    layer = _bn_relu(layer, **params)
    return layer


def add_output_layer(layer, **params):
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation("softmax")(layer)


def add_compile(model, **params):
    optimizer = Adam(
        learning_rate=params["learning_rate"], clipnorm=params.get("clipnorm", 1)
    )

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


def build_network(**params):
    inputs = tf.keras.Input(shape=params["input_shape"], dtype="float32", name="inputs")

    if params.get("is_regular_conv", False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        model = add_compile(model, **params)
    return model


def readDataWithoutLabels(input_files, frame_len):
    fs = 500
    num_leads = 12

    # __________________________________________________________________
    # Data from all mat files.
    # __________________________________________________________________
    num_files = len(input_files)
    X = []

    normalize = partial(processing.normalize_bound, lb=-1, ub=1)
    # Iterate over files.
    for i, f in enumerate(input_files):

        #         print('    {}/{}...'.format(i+1, num_files))
        # Creating temporary variables for current signal and label
        temp_x = np.zeros((1, frame_len, num_leads), dtype=np.float32)
        # Mat files. (ECG data)
        tmp_input_file = f
        data, header_data = load_challenge_data(tmp_input_file)

        # ___________________________________________________________________________
        # Reading Header data and processing it
        #

        # Header files. (ECG Labels)
        g = f.replace(".mat", ".hea")
        tmp_input_file = g

        # Read sampled frequency
        with open(tmp_input_file, "r") as f:
            first_line = f.readline()
            sampled_fs = int(first_line.split(" ")[2])
        # If sample frequency is not 500. Resample data
        if sampled_fs != fs:
            data = resample_poly(data, fs, sampled_fs, axis=1)

        # ___________________________________________________________________________
        # Reading Signal data and processing it
        #

        # If length of ecg signal is greater than the frame length just truncate it.
        if data.shape[1] > frame_len:
            data = data[:, :frame_len]

        extended_data = np.zeros((num_leads, frame_len))

        for j in range(num_leads):

            # If all values in a lead are not zero.
            if data[j, :].any():
                # Frame Normalization
                data[j, :] = np.squeeze(np.apply_along_axis(normalize, 0, data[j, :]))

            # padding zeros and bandpass filtering.
            data[j, :] = bandpass_filter(data[j, :])
            extended_data[j, :] = extend_ts(data[j, :], length=frame_len)

        temp_x = extended_data.T

        # ___________________________________________________________________________
        # Finalizing Labels and Signals into X and y
        #

        # Creating multiple Xs and ys for multi labelled files
        X.append(temp_x)

    return X


def save_challenge_predictions(output_directory, filenames, scores, labels, classes):

    for idx, file in enumerate(filenames):

        filename = os.path.basename(file)
        recording = os.path.splitext(filename)[0]
        new_file = filename.replace(".mat", ".csv")
        output_file = os.path.join(output_directory, new_file)

        # Include the filename as the recording number
        recording_string = "#{}".format(recording)
        class_string = ",".join(map(str, classes))
        label_string = ",".join(str(i) for i in labels[idx])
        score_string = ",".join(str(i) for i in scores[idx])

        with open(output_file, "w") as f:
            f.write(
                recording_string
                + "\n"
                + class_string
                + "\n"
                + label_string
                + "\n"
                + score_string
                + "\n"
            )


def save_original_labels(label_directory, filenames):
    for file in filenames:

        filename = os.path.basename(file)
        label_file = os.path.join(label_directory, filename.replace(".mat", ".hea"))
        shutil.copyfile(file.replace(".mat", ".hea"), label_file)


def equvialentClassesConversion(scored_classes, classes, labels):
    equivalent_classes_collection = [
        [713427006, 59118001],
        [284470004, 63593006],
        [427172004, 17338001],
    ]
    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes) > 1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels[:, representative_index] = np.any(
                labels[:, equivalent_indices], axis=1
            )
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
        del scored_classes[x]
    labels = np.delete(labels, remove_indices, axis=1)

    return scored_classes, classes, labels


def createBinarySVC(classes, max_cases_svc, frame_len):
    max_cases = max_cases_svc
    for idx, current_class in enumerate(classes):
        if current_class == 713427006:
            current_classes = [713427006, 59118001]
        elif current_class == 284470004:
            current_classes = [284470004, 63593006]
        elif current_class == 427172004:
            current_classes = [427172004, 17338001]
        else:
            current_classes = [current_class]
        # Remainder Classes
        remainder_classes = classes.copy()
        remainder_classes.remove(current_class)
        # Read All files with labels of current clas
        current_class_cases = [max_cases] * len(current_classes)
        input_files_current, _ = readFilesWithLabels(
            input_directory="data",
            classes=current_classes,
            classes_cases=current_class_cases,
        )
        print("input_files_current", len(input_files_current))
        # Claculate how many cases of current class
        current_class_cases = len(input_files_current)
        # Read All files with labels of remainder classes
        remainder_cases = [math.ceil(current_class_cases / (len(classes) - 1))] * (
            len(classes) - 1
        )
        input_files_remainder, _ = readFilesWithLabels(
            input_directory="data",
            classes=remainder_classes,
            classes_cases=remainder_cases,
        )
        print("input_files_remainder:", len(input_files_remainder))

        # Read data from files
        X_current = readDataWithoutLabels(input_files_current, frame_len)
        y_current = [1] * len(X_current)
        X_remainder = readDataWithoutLabels(input_files_remainder, frame_len)
        y_remainder = [0] * len(X_remainder)
        X = X_current + X_remainder
        y = y_current + y_remainder
        # Converting Python lists to final Training Array
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int)

        print(X.shape, y.shape)

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Get Features from feature Extractor
        feat_train = model_feat.predict(X_train, batch_size=bs)
        feat_train = feat_train.reshape(feat_train.shape[0], -1)
        feat_test = model_feat.predict(X_test, batch_size=bs)
        feat_test = feat_test.reshape(feat_test.shape[0], -1)
        print(feat_train.shape, feat_test.shape)

        # Train SVM
        print("Training SVM for Class", current_class)
        svm = SVC(kernel="linear")
        svm.probability = True
        svm.fit(feat_train, y_train)
        print("fitting done !!!")
        print("Train Score:", svm.score(feat_train, y_train))
        print("Test Score:", svm.score(feat_test, y_test))

        # save the model to disk
        filename = "svc_" + str(current_class) + ".sav"
        pickle.dump(svm, open(filename, "wb"))


if __name__ == "__main__":

    # Model Parameters

    bs = 32  # Batch size
    ep = 20  # epochs
    threshold = 0.5  # Threshold
    frame_len = 17920  # Frame Length
    steps = int(frame_len / 256)
    max_cases = 200  # Max cases for each class for CNN
    max_cases_svc = 200  # Max cases for each class for SVC
    params = {
        "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "conv_filter_length": 16,
        "conv_num_filters_start": 32,
        "conv_init": "he_normal",
        "conv_activation": "relu",
        "conv_dropout": 0.2,
        "conv_num_skip": 2,
        "conv_increase_channels_at": 4,
        "learning_rate": 0.001,
        "input_shape": [frame_len, 12],
        "num_categories": 24,
        "compile": True,
    }
    # __________________________________________________________________
    # All scored Classes
    # __________________________________________________________________
    scored_classes = {
        270492004: "IAVB",
        164889003: "AF",
        164890007: "AFL",
        426627000: "Brady",
        713427006: "CRBBB",
        713426002: "IRBBB",
        445118002: "LAnFB",
        39732003: "LAD",
        164909002: "LBBB",
        251146004: "LQRSV",
        698252002: "NSIVCB",
        10370003: "PR",
        284470004: "PAC",
        427172004: "PVC",
        164947007: "LPR",
        111975006: "LQT",
        164917005: "QAb",
        47665007: "RAD",
        59118001: "RBBB",
        427393009: "SA",
        426177001: "SB",
        426783006: "SNR",
        427084000: "STach",
        63593006: "SVPB",
        164934002: "TAb",
        59931005: "TInv",
        17338001: "VPB",
    }
    classes = sorted(scored_classes)
    # Define how many maximum cases for each class you want
    classes_cases = [max_cases] * len(classes)

    # Read All files with labels according to need as defined by classes_cases
    input_files, labels = readFilesWithLabels(
        input_directory="data", classes=classes, classes_cases=classes_cases
    )

    # Read data from files
    X, y, multi_labels = readData(
        input_files, frame_len, labels, classes, scored_classes
    )

    # Creating One Hot encoding scheme for given classes (27)
    n_features = 1
    n_labels = len(classes)
    categories = [range(n_labels)] * n_features
    onehot_encoder = OneHotEncoder(categories=categories, sparse=False)

    # Encoding labels
    y = onehot_encoder.fit_transform(y.reshape(-1, 1))

    # Convert Equivalent Classes Labels
    scored_classes, classes, y = equvialentClassesConversion(scored_classes, classes, y)

    # Creating One Hot encoding scheme for new classes (24)
    n_features = 1
    n_labels = len(classes)
    categories = [range(n_labels)] * n_features
    onehot_encoder = OneHotEncoder(categories=categories, sparse=False)

    # Split data into training and testing
    X_train, X_test, y_train, y_test, _, multi_labels_test, _, input_files_test = (
        train_test_split(
            X, y, multi_labels, input_files, stratify=y, test_size=0.2, random_state=42
        )
    )

    print(y_train.shape, y_test.shape)

    print(y_train.shape, y_test.shape)
    y_train = np.repeat(y_train[:, np.newaxis, :], steps, axis=1)
    y_test = np.repeat(y_test[:, np.newaxis, :], steps, axis=1)
    print(y_train.shape, y_test.shape)

    # gc.collect()
    del X, y, multi_labels, input_files

    # Create and Manage Directories for Results
    label_directory = os.path.join(os.getcwd(), "outputs", "labels")
    cnn_directory = os.path.join(os.getcwd(), "outputs", "cnn")
    shutil.rmtree(label_directory, ignore_errors=True)
    shutil.rmtree(cnn_directory, ignore_errors=True)
    Path(label_directory).mkdir(parents=True, exist_ok=True)
    Path(cnn_directory).mkdir(parents=True, exist_ok=True)

    # Save Test Labels Seperately for score Calculation
    save_original_labels(label_directory=label_directory, filenames=input_files_test)

    # Create Stats
    stats_new = np.zeros((1, 7))

    # __________________________________________________________________
    # Create Models
    # __________________________________________________________________

    # Define CNN model architecture
    # model_cnn=create_model(frame_len,num_classes)
    model_cnn = build_network(**params)
    model_path_cnn = "cnn_model3.h5"

    stopping = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    reduce_lr = ReduceLROnPlateau(
        factor=0.1, patience=2, min_lr=params["learning_rate"] * 0.001
    )

    checkpointer = ModelCheckpoint(
        monitor="val_loss", mode="min", filepath=model_path_cnn, save_best_only=True
    )

    print("------------------------------------------------------------------------")
    print("Training CNN Model...")
    try:
        model_cnn.load_weights(model_path_cnn)
    except:
        pass

    history = model_cnn.fit(
        X_train,
        y_train,
        batch_size=bs,
        epochs=ep,
        validation_data=(X_test, y_test),
        callbacks=[checkpointer, stopping, reduce_lr],
    )  # starts training

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))
    fig = plt.figure()
    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig("accuracy_cnn.png")
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig("loss_cnn.png")
    plt.close(fig)

    del X_train, y_train, multi_labels_test

    # Load best epoch weights.
    model_cnn.load_weights(model_path_cnn)

    print("------------------------------------------------------------------------")
    print("Define Feature Extractor and create Binary SVCs")
    model_feat = Model(
        inputs=model_cnn.input,
        outputs=model_cnn.get_layer("global_max_pooling1d").output,
    )
    createBinarySVC(classes, max_cases_svc)

    print("------------------------------------------------------------------------")
    print("Testing Through SVC")

    # Get Features from feature Extractor
    feat_test = model_feat.predict(X_test, batch_size=bs)
    y_test_pred = []
    y_test_pred_score = []
    for idx, current_class in enumerate(classes):

        # load the model from disk
        filename = "svc_" + str(current_class) + ".sav"
        loaded_model = pickle.load(open(filename, "rb"))
        # Make predictions
        y_test_pred.append(loaded_model.predict(feat_test))
        y_test_pred_score.append(loaded_model.predict_proba(feat_test)[:, -1])

    y_test_pred = np.asarray(y_test_pred, dtype=np.int).T
    y_test_pred_score = np.asarray(y_test_pred_score, dtype=np.float32).T

    # Save Model Predictions
    save_challenge_predictions(
        output_directory=cnn_directory,
        filenames=input_files_test,
        scores=y_test_pred_score,
        labels=y_test_pred,
        classes=sorted(scored_classes),
    )

    # Compute new Scores
    (
        auroc,
        auprc,
        accuracy1,
        f_measure1,
        f_beta_measure1,
        g_beta_measure1,
        challenge_metric,
    ) = evaluate_12ECG_score(
        label_directory=label_directory, output_directory=cnn_directory
    )
    stats_new[0] = [
        auroc,
        auprc,
        accuracy1,
        f_measure1,
        f_beta_measure1,
        g_beta_measure1,
        challenge_metric,
    ]

    # Collect unused variables and delete unused variables
    gc.collect()
    del X_test, y_test

    # Save stats

    df = pd.DataFrame(
        stats_new,
        columns=[
            "auroc",
            "auprc",
            "accuracy",
            "f_measure",
            "f_beta_measure",
            "g_beta_measure",
            "challenge_metric",
        ],
        index=["CNN"],
    )
    df.to_csv("stats_new.csv")
