import hashlib
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pyedflib
import requests
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

home = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/edfs/home/homepap-home-"
lab_full = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/edfs/lab/full/homepap-lab-full-"
split = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/edfs/lab/split/homepap-lab-split-"

annt_home = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/annotations-events-nsrr/home/homepap-home-"
annt_lab_full = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/annotations-events-profusion/lab/full/homepap-lab-full-"
annt_split = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/polysomnography/annotations-events-profusion/lab/split/homepap-lab-split-"

# servers = [home, lab_full, split]
servers = [(lab_full, annt_lab_full), (split, annt_split)]


descriptor = "https://anon.erda.au.dk/share_redirect/DCuFnOpr1n/datasets/homepap-baseline-dataset-0.2.0.csv"
records = pd.read_csv(descriptor)["nsrrid"].values


def verify_xml(file_path):
    """
    Verifies the integrity of an XML file by attempting to parse it with ElementTree.
    If the file is corrupt, ElementTree will raise an exception.
    """
    try:
        tree = ET.parse(file_path).findall("SleepStages")[0]
    except ET.ParseError as e:
        return False

    return True


def verify_edf(file_path):
    """
    Verifies the integrity of an EDF file by attempting to open it with pyedflib.
    If the file is corrupt, pyedflib will raise an exception.
    """
    try:
        f = pyedflib.EdfReader(file_path)
    except Exception as e:
        return False

    return True


def download_file_in_parts(url, file_path):
    response = requests.head(url)
    file_size = int(response.headers["content-length"])

    # log the file_size in mb
    logger.info(f"Downloading {file_path} of size {file_size / 1024 / 1024:.2f} MB")

    chunk_size = 1024 * 1024  # 1 MB
    num_chunks = file_size // chunk_size

    with open(file_path, "wb") as f:
        for i in range(num_chunks + 1):
            start = i * chunk_size
            end = start + chunk_size - 1
            if end > file_size - 1:
                end = file_size - 1

            headers = {"Range": f"bytes={start}-{end}"}
            response = requests.get(url, headers=headers)
            f.write(response.content)


def download_file(file_path, server_name, record_id, url):
    # check that the file does not already exist
    if os.path.exists(file_path):
        if file_path.endswith(".edf") and verify_edf(file_path):
            logger.info(f"Skipping edf record {record_id} from server {server_name}")
            return None
        elif file_path.endswith(".xml") and verify_xml(file_path):
            logger.info(f"Skipping xml record {record_id} from server {server_name}")
            return None

        else:
            os.remove(file_path)

    response = requests.get(url, allow_redirects=True)

    if response.ok:

        download_file_in_parts(url, file_path)

        if file_path.endswith(".edf") and verify_edf(file_path):
            logger.info(f"Downloaded edf record {record_id} from server {server_name}")
            time.sleep(10)
            return True
        elif file_path.endswith(".xml") and verify_xml(file_path):
            logger.info(f"Downloaded xml record {record_id} from server {server_name}")
            time.sleep(10)
            return True

        else:
            os.remove(file_path)
            logger.error(
                f"Corrupted {file_path[-3:]} record {record_id} from server {server_name}"
            )
            time.sleep(10)
            return False
    else:
        logger.error(
            f"Record {file_path[-3:]} {record_id} not in server: {server_name}"
        )
        time.sleep(10)
        return True


def download(record_id, out_path):
    """
    Download file 'file_name' and validate sha256 checksum against 'sha256'.
    Saves the downloaded file to 'out_path'.
    If the file already exists, and have a valid sha256, the download is skipped.
    """

    # try to download the file from the servers
    res_edf, res_annt = [], []
    for server, annt_server in servers:
        server_name = (
            "home"
            if server == home
            else "lab-full" if server == lab_full else "lab-split"
        )
        file_path = out_path + server_name + "-" + record_id + ".edf"
        annt_path = out_path + server_name + "-" + record_id + "-profusion.xml"

        # res_edf.append( download_file(file_path, server_name, record_id, server + record_id + ".edf") )
        res_annt.append(
            download_file(
                annt_path,
                server_name,
                record_id,
                annt_server + record_id + "-profusion.xml",
            )
        )
        res_edf.append(True)

    # se c'`e almeno un file corrotto ritorna l' id del record

    if not all(res_edf) or not all(res_annt):
        return record_id

    logger.error(f"Could not download record {record_id}")

    return record_id


def download_dataset(out_dataset_folder, my_records):

    while True:
        records_list = Parallel(n_jobs=5, backend="threading")(
            delayed(download)(record, out_dataset_folder) for record in my_records
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

    edf_files = [
        file
        for file in os.listdir("/home/guido/shared/hpap/download/")
        if file.endswith(".edf") and file.startswith("lab")
    ]
    xml_files = [
        file
        for file in os.listdir("/home/guido/shared/hpap/download/")
        if file.endswith("-profusion.xml") and file.startswith("lab")
    ]

    print(len(edf_files), len(xml_files))

    # print the records of edf which have no xml associated

    for edf in edf_files:
        record_id = (
            edf.replace("lab-full-", "").replace("lab-split-", "").replace(".edf", "")
        )
        if not any(record_id in xml for xml in xml_files):
            print(record_id)
            # print also the server from which the record was downloaded
            print("lab-full" if "lab-full" in edf else "lab-split")
            # print also the url from which to download the xml
            print(
                annt_lab_full + record_id + "-profusion.xml"
                if "lab-full" in edf
                else annt_split + record_id + "-profusion.xml"
            )

    for edf, xml in tqdm(zip(edf_files, xml_files)):

        if not verify_edf(os.path.join("/home/guido/shared/hpap/download/", edf)):
            logger.error(f"Corrupted edf record {edf}")
            os.remove(os.path.join("/home/guido/shared/hpap/download/", edf))

            server_name = "lab-full" if "lab-full" in edf else "lab-split"
            server_url = lab_full if server_name == "lab-full" else split
            url = server_url + edf.replace(server_name + "-", "")
            record_id = edf.replace(server_name + "-", "").replace(".edf", "")

            # download the record again
            download_file(
                os.path.join("/home/guido/shared/hpap/download/", edf),
                server_name,
                record_id,
                url,
            )

        if not verify_xml(os.path.join("/home/guido/shared/hpap/download/", xml)):
            logger.error(f"Corrupted xml record {xml}")
            os.remove(os.path.join("/home/guido/shared/hpap/download/", xml))

            server_name = "lab-full" if "lab-full" in xml else "lab-split"
            server_url = lab_full if server_name == "lab-full" else split
            url = server_url + xml.replace(server_name + "-", "")
            record_id = xml.replace(server_name + "-", "").replace("-profusion.xml", "")

            # download the record again
            download_file(
                os.path.join("/home/guido/shared/hpap/download/", xml),
                server_name,
                record_id,
                url,
            )

    """
    # dividi il numero totale di record per il numero totale di nodi e prendi quelli relativi al tuo nodo
    num_records = len(records)
    records_per_node = num_records // args.total_nodes

    logger.info(
        f"Node {args.node_id} will download records from {args.node_id * records_per_node} to {(args.node_id + 1) * records_per_node}"
    )

    my_records = list(
        range(args.node_id * records_per_node, (args.node_id + 1) * records_per_node)
    )

    # update the dataframe with the records to download
    my_records = list(records[my_records])
    my_records = [str(record) for record in my_records]

    download_dataset("/home/guido/shared/hpap/download/", my_records= my_records)
    """
