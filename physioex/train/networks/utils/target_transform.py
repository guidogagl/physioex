def get_mid_label(labels):

    sequence_length = labels.size(0)
    mid_sequence = int((sequence_length - 1) / 2)

    return labels[mid_sequence]
