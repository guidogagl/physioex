import argparse
import physioex.train as train

def main():
    parser = argparse.ArgumentParser(description='Training script')
    
    # experiment arguments
    parser.add_argument('-e', '--experiment', default='chambon2018', help='Specify the experiment')
    parser.add_argument('-s', '--similarity', default=False, help='Specify the experiment')
    
    # dataset args
    parser.add_argument('-d', '--dataset', default='SleepPhysionet', help='Specify the dataset')
    parser.add_argument('-v', '--version', default='2018', help='Specify the dataset')
    parser.add_argument('-c', '--use_cache', default=True, help='Specify the dataset')
    
    # sequence
    parser.add_argument('-sl', '--sequence_lenght', default=3, help='Specify the dataset')
    
    # trainer
    parser.add_argument('-me', '--max_epoch', default=20, help='Specify the dataset')
    parser.add_argument('-vci', '--val_check_interval', default=300, help='Specify the dataset')
    parser.add_argument('-bs', '--batch_size', default=32, help='Specify the dataset')
    
    # Aggiungi altri argomenti di default o specifici di PyTorch Lightning

    args = parser.parse_args()
    
    experiment = getattr(train, args.experiment)
    
    experiment(args.similarity, args.dataset, args.version, args.use_cache, args.sequence_lenght, args.max_epoch, args.val_check_interval, args.batch_size)

    
if __name__ == '__main__':
    main()
