import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import requests
from joblib import Parallel, delayed
from loguru import logger

server_url = "https://physionet.org/files/challenge-2018/1.0.0/"

checksum_path = os.path.join(os.path.dirname(__file__), "checksum.csv")
checksum_df = pd.read_csv(checksum_path, header=None, sep=" ")
checksum_df.columns = ["sha256", "record"]


def get_sha256(local_path):
    """
    Computes the sha256 of a local file.

    :param local_path: string/path to file
    :return: sha256 checksum
    """
    file_hash = hashlib.sha256()
    with open(local_path, "rb") as in_f:
        for chunk in iter(lambda: in_f.read(512 * file_hash.block_size), b""):
            file_hash.update(chunk)
    return file_hash


def validate_sha256(local_path, sha256):
    """
    Computes the sha256 checksum of a file and compares it to the passed sha256 hexdigest checksum
    """
    file_hash = get_sha256(local_path)
    return file_hash.hexdigest() == sha256


def download_and_validate(download_url, sha256, out_path, record_id):
    """
    Download file 'file_name' and validate sha256 checksum against 'sha256'.
    Saves the downloaded file to 'out_path'.
    If the file already exists, and have a valid sha256, the download is skipped.
    """
    if os.path.exists(out_path):
        if validate_sha256(out_path, sha256):
            logger.info("... skipping (already downloaded with valid sha256)")
            return
        else:
            logger.info("... File exists, but invalid SHA256, re-downloading")

    response = requests.get(download_url, allow_redirects=True)
    if response.ok:
        with open(out_path, "wb") as out_f:
            out_f.write(response.content)
    else:
        logger.error(
            "Could not download file from URL {}. "
            "Received HTTP response with status code {}".format(
                download_url, response.status_code
            )
        )
        time.sleep(10)
        return record_id

    if not validate_sha256(out_path, sha256):
        os.remove(out_path)
        logger.warning(
            f"Invalid sha256 for file at {download_url} " f"(please restart download)"
        )
        time.sleep(10)
        return record_id

    logger.info(f"Downloaded file {record_id} with valid sha256")
    # Sleep for a bit to avoid hammering the server
    time.sleep(10)
    return None


def get_out_path(file_name, out_dataset_folder):
    out_file_path = os.path.join(out_dataset_folder, file_name)
    out_file_path = Path(out_file_path)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    return out_file_path


def download_dataset(out_dataset_folder, checksum_df=checksum_df):
    """
    Download a dataset into 'out_dataset_folder' by fetching files at URL 'server_url' according to
    the list of checksums and filenames in file 'checksums_path'. Only downloads the N_first subject folders
    if 'N_first' is specified.

    'paths_func' should be a callable of signature func(file_name, server_url, out_dataset_folder) which returns:
        1) download_url (path to fetch file from on remote system)
        2) out_file_path (path to store file on local system)
    """
    checksums, file_names = list(checksum_df.sha256.values), list(
        checksum_df.record.values
    )
    records_list = list(range(len(checksums)))
    total_files = len(records_list)

    while True:
        records_list = Parallel(n_jobs=5, backend="threading")(
            delayed(download_and_validate)(
                server_url + file_names[record_id],
                checksums[records_list[record_id]],
                get_out_path(file_names[record_id], out_dataset_folder),
                record_id,
            )
            for record_id in records_list
        )

        records_list = [
            record_id for record_id in records_list if record_id is not None
        ]
        if len(records_list) == 0:
            break


import argparse

if __name__ == "__main__":

    # prendi l' identificativo del nodo e il numero totale di nodi
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--total_nodes", type=int, default=1)
    args = parser.parse_args()

    # dividi il numero totale di record per il numero totale di nodi e prendi quelli relativi al tuo nodo
    num_records = len(checksum_df)
    records_per_node = num_records // args.total_nodes

    logger.info(
        f"Node {args.node_id} will download records from {args.node_id * records_per_node} to {(args.node_id + 1) * records_per_node}"
    )

    my_records = list(
        range(args.node_id * records_per_node, (args.node_id + 1) * records_per_node)
    )

    # update the dataframe with the records to download
    checksum_df = checksum_df.iloc[my_records]

    download_dataset("/home/guido/shared/physio2018/download/", checksum_df=checksum_df)
