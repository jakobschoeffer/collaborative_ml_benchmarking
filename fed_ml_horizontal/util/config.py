import glob
import os

from box import Box


def read_config(additional_config=None):

    config = Box()
    config_files = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "config", "*.yml")
    )

    for file in config_files:
        config.merge_update(Box.from_yaml(filename=file))

    if additional_config:
        config.merge_update(Box.from_yaml(filename=additional_config))

    return config
