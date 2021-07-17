import importlib
import logging

from fed_ml_horizontal.util.config import read_config
from fed_ml_horizontal.util.logging import setup_logging


def run():
    """runs pipeline"""
    # read config and create logger
    config = read_config()
    setup_logging(config)

    logging.info(str(config))
    logging.info("Read config and wrote config to log file")

    # execute all steps of the pipeline
    steps = config.project.steps
    logging.info(f"Running {len(steps)} steps from {steps[0]} to {steps[-1]}")

    for step in steps:
        mod_name, func_name = step.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        func(config)


if __name__ == "__main__":
    run()
