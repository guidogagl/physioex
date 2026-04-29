from physioex.data.dataset import get_dataloaders, PhysioExDataset
import time
import torch


def main():
    dataset = PhysioExDataset(datasets=["hmc"])

    print("SleepEDF N subjects: ", dataset.get_n_subjects())
    table = dataset.get_table(0)

    print("N° epochs: ", table["num_windows"].values.sum())

    dataset = PhysioExDataset(datasets=["sleepedf"])

    print("SleepEDF N subjects: ", dataset.get_n_subjects())
    table = dataset.get_table(0)

    print("N° epochs: ", table["num_windows"].values.sum())

    dataset = PhysioExDataset(datasets=["PD/PD", "PD/HOA"])

    print("PD N subjects: ", dataset.get_n_subjects())

    tables = [dataset.get_table(i) for i in [0, 1]]
    num_epochs = [dataset["num_windows"].values.sum() for dataset in tables]
    num_epochs = num_epochs[0] + num_epochs[1]

    print("N° epochs: ", num_epochs)

    dataset = PhysioExDataset(datasets=["AD/AD", "AD/HOA"])

    print("AD N subjects: ", dataset.get_n_subjects())

    tables = [dataset.get_table(i) for i in [0, 1]]
    num_epochs = [dataset["num_windows"].values.sum() for dataset in tables]
    num_epochs = num_epochs[0] + num_epochs[1]

    print("N° epochs: ", num_epochs)

    exit()

    dataset = PhysioExDataset(datasets=["AD/AD", "AD/HC"])

    num_workers = 16

    # Iterate through the training dataloader
    # and compute the epoch fetch time
    print("Testing train_loader...")
    start_time = time.time()
    for batch_idx, (signals, labels) in enumerate(train_loader):
        pass  # Simulate training step
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken to fetch one epoch of training data: {epoch_time:.2f} seconds")

    # Similarly, you can test valid_loader and test_loader if needed
    print("Testing valid_loader...")
    start_time = time.time()
    night_lenght = []
    for batch_idx, (signals, labels) in enumerate(valid_loader):
        labels = labels[labels != -1]
        night_lenght.append(labels.view(-1).size(-1))
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken to fetch one epoch of validation data: {epoch_time:.2f} seconds")
    print(f"Night lengths: { torch.tensor( night_lenght ).mean() * 2 / 60 :.2f} hours")

    print("Testing test_loader...")
    start_time = time.time()
    night_lenght = []
    for batch_idx, (signals, labels) in enumerate(test_loader):
        # remove -1 labels
        labels = labels[labels != -1]

        night_lenght.append(labels.view(-1).size(-1))
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken to fetch one epoch of testing data: {epoch_time:.2f} seconds")
    print(f"Night lengths: { torch.tensor( night_lenght ).mean() * 2 / 60 :.2f} hours")


if __name__ == "__main__":
    main()
