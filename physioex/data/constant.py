from pathlib import Path

data_folder = str(Path.home())


def get_data_folder():
    return data_folder


def set_data_folder(new_data_folder):
    global data_folder
    data_folder = new_data_folder
