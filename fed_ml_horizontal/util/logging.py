import logging
import os
import sys
import time

import tensorflow as tf


def setup_logging(config):
    # create logging directory if not exists
    if not os.path.isdir(config.project.log_path):
        os.makedirs(config.project.log_path)

    # Remove all handlers associated with the root logger object.
    # (necessary when using this in jupyter notebook since calling basicConfig
    # only adds handlers and does not replace existing handlers)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # set up logging to file and console
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfileprefix = config.project.log_file_prefix

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler(
        os.path.join(
            config.project.log_path,
            f"{logfileprefix}_{timestamp}.log",
        )
    )
    fileHandler.setFormatter(logFormatter)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        handlers=[fileHandler, consoleHandler],
    )
