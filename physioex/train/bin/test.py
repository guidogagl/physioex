from physioex.train.bin.parser import PhysioExParser

def test_script():

    parser = PhysioExParser.test_parser()

    # check if we are running in fast mode or not
    if parser["fast"]:
        from physioex.train.utils.fast_test import test
    else:
        from physioex.train.utils.test import test
    
    datamodule_kwargs = {
        "selected_channels": parser["selected_channels"],
        "sequence_length": parser["sequence_length"],
        "target_transform": parser["target_transform"],
        "preprocessing": parser["preprocessing"],
        "task": parser["model_task"],
        "data_folder": parser["data_folder"],
        "num_workers": parser["num_workers"],
    }

    test(
        datasets=parser["datasets"],
        datamodule_kwargs=datamodule_kwargs,
        model=None,
        fold=parser["fold"],
        model_class=parser["model"],
        model_config=parser["model_kwargs"],
        batch_size=parser["batch_size"],
        hpc=parser["hpc"],
        num_nodes=parser["num_nodes"],
        checkpoint_path=parser["checkpoint_path"],
        results_path=parser["results_path"],
        aggregate_datasets=parser["aggregate"],
    )


if __name__ == "__main__":
    test_script()