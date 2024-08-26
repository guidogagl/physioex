from pathlib import Path
from itertools import combinations
import subprocess
import time

from loguru import logger

if __name__ == "__main__":
    
    multi_source_module = False
    from itertools import combinations
    import subprocess
    import time
    
    datasets = ["mass", "hmc", "dcsm", "mesa" ] #"mros"]
    
    train_datasets = [
        combo
        for r in range(1, len(datasets) + 1)
        for combo in combinations(datasets, r)
    ]
    
    # add empty combination at the beginning
    
    #train_datasets.insert(0, [])
    
    print(train_datasets)
    max_parallel_screens = 5  # Numero massimo di screen da eseguire contemporaneamente
    screen_sessions = []

    for train_dataset in train_datasets:
        print(train_dataset)
        screen_name = "train_" + "_".join(train_dataset)
        log_dir = f"log/msd/fine_tuned_model/k={len(train_dataset)}/{'_'.join(train_dataset)}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{log_dir}/output.log"
        ckp_path = f"models/multi-source-domain/finetunedmodel/k={len(train_dataset)}/{'_'.join(train_dataset)}/"

        train_dataset = " ".join(train_dataset)
        command = (
            f'screen -dmS {screen_name} bash -c "'
            f'source /home/guido/miniconda3/etc/profile.d/conda.sh && '
            f'conda activate physioex && '
            f'python physioex/train/finetuner.py '
            f'--train_datasets {train_dataset} '
            f'--ckp_path {ckp_path} '
            f'--batch_size 64 '
            f'--random_fold '
            f'--data_folder /mnt/guido-data/ '
            f'--max_epoch 10 '
            f'--val_check_interval 20 '
            f' > {log_file} 2>&1"' 
        )

        subprocess.run(command, shell=True)
        screen_sessions.append(screen_name)
        logger.info(f"Launched main in screen session: {screen_name}")

        # Controlla se il numero massimo di screen attive è stato raggiunto
        while len(screen_sessions) >= max_parallel_screens:
            for session in screen_sessions:
                # Controlla se la sessione è ancora attiva
                result = subprocess.run(
                    f"screen -list | grep {session}", shell=True, stdout=subprocess.PIPE
                )
                if not result.stdout:
                    screen_sessions.remove(session)
                    break  # Una sessione è terminata, quindi possiamo avviarne un'altra
            if len(screen_sessions) >= max_parallel_screens:
                time.sleep(60)  # Aspetta 60 secondi prima di controllare di nuovo

    # Aspetta che tutte le sessioni attive terminino
    all_done = False
    while not all_done:
        all_done = True
        for session in screen_sessions:
            # Controlla se la sessione è ancora attiva
            result = subprocess.run(
                f"screen -list | grep {session}", shell=True, stdout=subprocess.PIPE
            )
            if result.stdout:
                all_done = False
                break  # Una sessione è ancora attiva, quindi aspetta e poi controlla di nuovo
        if not all_done:
            time.sleep(60)  # Aspetta 60 secondi prima di controllare di nuovo

    logger.info(f"All sessions completed.")