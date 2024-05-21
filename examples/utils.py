import os
import subprocess


def set_root():
    # Ottieni il percorso corrente
    current_path = os.getcwd()

    # Esegui il comando git per trovare la root del repository
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=current_path
    )

    # Decodifica l'output per ottenere una stringa
    git_root_path = git_root.decode("utf-8").strip()

    # Cambia la directory corrente alla root del repository git
    os.chdir(git_root_path)

    # Stampa il nuovo percorso corrente
    print(f"Current working directory: {os.getcwd()}")

    # Aggiungi il percorso alla variabile d'ambiente PATH
    os.environ["PATH"] += os.pathsep + git_root_path
