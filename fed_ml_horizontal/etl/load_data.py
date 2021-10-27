import logging
import os
import shutil
import urllib.request
import zipfile


def load_data(config):
    """Load data from url.

    Args:
        config (Box): config object with project and scenario specifications
    """
    logging.info(
        f"Downloading and unzipping data from {config.project.data_url} to {config.project.data_path}"
    )
    download_and_unzip(
        data_url=config.project.data_url, data_path=config.project.data_path
    )


def download_and_unzip(data_url: str, data_path: str):
    """Download and unzip data.

    Args:
        data_url (str): data url defined in config object
        data_path (str): data path defined in config object
    """
    # remove data_path if already existing
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)

    os.makedirs(data_path)

    # download zip file
    data_path_zip = data_path + ".zip"
    urllib.request.urlretrieve(data_url, data_path_zip)

    # unzip at data_path
    zip_ref = zipfile.ZipFile(data_path_zip, "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    logging.info(f"Downloaded and unzipped data from {data_url} to {data_path}")
    logging.info(
        "From now on, the data is stored in the tmp folder and the fed_ml_horizontal.etl.load_data.load_data step can be commented out in config/project.yml"
    )
