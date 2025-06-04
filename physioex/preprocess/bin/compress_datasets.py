import argparse
import os
from physioex.data import PhysioExDataset, PhysioExDataModule
import torch
from tqdm import tqdm
import os

from lightning.pytorch import seed_everything

def compress_eval_dataset(loader, output_path, loader_type = "eval"):

    inputs, targets = [], []
    for batch in tqdm(loader, desc=f"Compressing {loader_type} dataset"):
        X, y, _, _ = batch

        if args.dataset == "shhs":
            X = X.to(torch.float16)
        else:
            X = X.float()

        y = y.long()

        # X shape 1, night_lenght, channels, input_shape

        # get if nan or inf values are present
        if torch.isnan(X).any() or torch.isinf(X).any():
            print(f"Error: NaN or Inf values found in {loader_type} dataset")
            exit(1)

        X = X.squeeze(0)
        y = y.squeeze(0)

        inputs.append(X)
        targets.append(y)    
        
    inputs = tuple( inputs )
    targets = tuple( targets )
    
    #print(f"Inputs shape: {len(inputs).shape}, Targets shape: {targets.shape}")
    
    output_path = os.path.join(output_folder, f"{loader_type}_dataset.pt")
    
    print(f"Saving {loader_type} dataset to {output_path}")
    torch.save((inputs, targets), output_path)


def compress_dataset(loader, output_path, loader_type = "train"):            
    # iterate over the loader, compress the X tensor from torch.float32 to torch.float16
    # save the compressed tensor to a file
    
    inputs, targets = [], []
    for batch in tqdm(loader, desc=f"Compressing {loader_type} dataset"):
        X, y, _, _ = batch

        if args.dataset == "shhs":
            X = X.to(torch.float16)
        else:
            X = X.float()

        y = y.long()

        # X shape 1, night_lenght, channels, input_shape

        # get if nan or inf values are present
        if torch.isnan(X).any() or torch.isinf(X).any():
            print(f"Error: NaN or Inf values found in {loader_type} dataset")
            exit(1)

        X = X.squeeze(0)
        y = y.squeeze(0)

        inputs.append(X)
        targets.append(y)    
        
    inputs = torch.cat(inputs, dim = 0)
    targets = torch.cat(targets, dim = 0)
    
    print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    
    output_path = os.path.join(output_folder, f"{loader_type}_dataset.pt")
    
    print(f"Saving {loader_type} dataset to {output_path}")
    torch.save((inputs, targets), output_path)


if __name__ == "__main__":
    
    seed_everything(42, workers=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compress datasets")
    parser.add_argument(
        "--data_folder", default="/mnt/guido-data/", type=str, help="data folder"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Specify the datasets list to train the model on. Expected type: list. Default: ['sleepedf']",
        type=str,
        default="sleepedf",
    )
    
    parser.add_argument(
        "--output_folder", default="./.tmp/", type=str, help="output folder"
    )

    args = parser.parse_args()

    for preprocessing in ["raw", "xsleepnet"]:

        print( "Creating PhysioExDataModule...")
        data = PhysioExDataModule(
            datasets = [args.dataset],
            batch_size = 1,
            preprocessing = preprocessing,
            selected_channels = ["EEG", "EOG", "EMG"],
            sequence_length = -1,
            data_folder = args.data_folder,
            num_workers= os.cpu_count(),
        )

        output_folder = os.path.join(args.output_folder, args.dataset, preprocessing) 
        os.makedirs(output_folder, exist_ok=True)
        
        # get the dataloaders
        train_loader = data.train_dataloader()
        eval_loader = data.val_dataloader()
        test_loader = data.test_dataloader()
        
        
        print("Compressing datasets...")
        # compress the train, eval and test datasets
        compress_eval_dataset(eval_loader, output_folder, "eval")
        compress_eval_dataset(test_loader, output_folder, "test")
        
        compress_dataset(train_loader, output_folder, "train")

        print(f"Datasets compressed and saved to {output_folder}")
    print("Done!")