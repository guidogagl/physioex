import argparse
import physioex.train as train

def main():
    parser = argparse.ArgumentParser(description='Training script')
    
    # experiment arguments
    parser.add_argument('-e', '--experiment', default='chambon2018', type = str, help='Specify the experiment to run. Expected type: str. Default: "chambon2018"')
    parser.add_argument('-s', '--similarity', default=False, type = bool, help='Specify whether to use similarity in the model. Expected type: bool. Default: False')
    
    # dataset args
    parser.add_argument('-d', '--dataset', default='SleepPhysionet', type = str, help='Specify the dataset to use. Expected type: str. Default: "SleepPhysionet"')
    parser.add_argument('-v', '--version', default='2018', type = str, help='Specify the version of the dataset. Expected type: str. Default: "2018"')
    parser.add_argument('-c', '--use_cache', default=True, type = bool, help='Specify whether to use cache for the dataset. Expected type: bool. Default: True')
    
    # sequence
    parser.add_argument('-sl', '--sequence_lenght', default=3, type=int, help='Specify the sequence length for the model. Expected type: int. Default: 3')
    
    # trainer
    parser.add_argument('-me', '--max_epoch', default=20, type = int, help='Specify the maximum number of epochs for training. Expected type: int. Default: 20')
    parser.add_argument('-vci', '--val_check_interval', default=300, type = int, help='Specify the validation check interval during training. Expected type: int. Default: 300')
    parser.add_argument('-bs', '--batch_size', default=32, type = int, help='Specify the batch size for training. Expected type: int. Default: 32')
    
    # Aggiungi altri argomenti di default o specifici di PyTorch Lightning

    args = parser.parse_args()
    
    experiment = getattr(train, args.experiment)
    
    experiment(args.similarity, args.dataset, args.version, args.use_cache, int(args.sequence_lenght), int(args.max_epoch), int(args.val_check_interval), int(args.batch_size))

    
if __name__ == '__main__':
    main()
