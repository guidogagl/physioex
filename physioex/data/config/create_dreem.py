import random

import yaml

num_subjects = [25, 56]


def generate_splits(num_subjects, num_folds):
    subjects = list(range(num_subjects))
    random.shuffle(subjects)

    folds = dict()

    for i in range(num_folds):

        splits = {"train": [], "valid": [], "test": []}
        fold_size = num_subjects // num_folds
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        test_subjects = subjects[start_idx:end_idx]
        train_subjects = subjects.copy()

        splits["test"] = list(test_subjects)

        for subj in test_subjects:
            train_subjects.remove(subj)

        valid_size = int(0.2 * len(train_subjects))

        valid_subjects = train_subjects[:valid_size]
        train_subjects = train_subjects[valid_size:]

        splits["valid"] = list(valid_subjects)
        splits["train"] = list(train_subjects)

        folds["fold_%d" % i] = splits

    return folds


def create_yaml_file():
    num_folds = 10
    data = {"dodh": {}, "dodo": {}}

    for i, dataset in enumerate(data):
        splits = generate_splits(num_subjects[i], num_folds)
        data[dataset] = splits

    with open("dreem.yaml", "w") as file:
        yaml.dump(data, file)


if __name__ == "__main__":
    create_yaml_file()
