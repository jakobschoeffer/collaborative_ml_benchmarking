import logging
import os
import time
import traceback
from contextlib import redirect_stdout

import pandas as pd
import tensorflow as tf

import fed_ml_horizontal.benchmarking.data_splitting as splitting
from fed_ml_horizontal.benchmarking.all_data_model import run_all_data_model
from fed_ml_horizontal.benchmarking.federated_learning import run_federated_model
from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.one_model_per_client import run_one_model_per_client


def run_scenarios(config):
    """Main function that iterates over all scenarios defined in the config file.
    For each scenario the federated, all_data and one model per client setting is performed.

    Args:
        config (Box): config object with project and scenario specifications
    """
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        logging.warn(
            "Could not set memory growth: Invalid device or cannot modify virtual devices once initialized."
        )
        pass

    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1500)]
    #         )
    #         logical_gpus = tf.config.list_logical_devices("GPU")
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    output_path_for_session = os.path.join(
        config.project.output_path, f"HFL_KIT_PITTING_CLASS_{current_datetime}"
    )
    os.makedirs(output_path_for_session)
    config.to_yaml(os.path.join(output_path_for_session, "config.yml"))

    for scenario in config.scenarios:
        try:
            current_datetime = time.strftime("%Y%m%d-%H%M%S")
            output_path_for_scenario = os.path.join(
                output_path_for_session, scenario + "_" + current_datetime
            )
            os.makedirs(output_path_for_scenario)
            scenario_config = config.scenarios[scenario]
            logging.info(
                f"Running {scenario} with configuration {scenario_config}. Output path: {output_path_for_scenario}"
            )

            (
                client_dataset_dict,
                all_images_path,
            ) = splitting.create_client_dataset_dict(
                data_path=config.project.data_path,
                output_dir=output_path_for_scenario,
                total_share=scenario_config.total_share,
                test_share=scenario_config.test_share,
                valid_share=scenario_config.valid_share,
                client_split_dict=scenario_config.clients,
            )

            # save model summary
            with open(
                os.path.join(output_path_for_scenario, "model_summary.txt"), "w"
            ) as f:
                with redirect_stdout(f):
                    create_my_model().summary()

            logging.info("Start federated model")
            results_fl = run_federated_model(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                max_num_epochs=scenario_config.max_num_epochs,
                learning_rate=scenario_config.learning_rate,
                early_stopping_patience=scenario_config.early_stopping_patience,
                early_stopping_monitor=scenario_config.early_stopping_monitor,
            )
            logging.info("Finished federated model")

            logging.info("Start all data model")
            results_all_data = run_all_data_model(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                max_num_epochs=scenario_config.max_num_epochs,
                learning_rate=scenario_config.learning_rate,
                early_stopping_patience=scenario_config.early_stopping_patience,
                early_stopping_monitor=scenario_config.early_stopping_monitor,
            )
            logging.info("Finished all data model")

            logging.info("Start one model per client")
            results_one_model_per_client = run_one_model_per_client(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                max_num_epochs=scenario_config.max_num_epochs,
                learning_rate=scenario_config.learning_rate,
                early_stopping_patience=scenario_config.early_stopping_patience,
                early_stopping_monitor=scenario_config.early_stopping_monitor,
            )
            logging.info("Finished one model per client")
        except Exception as e:
            logging.error(f"An exception occured trying to run {scenario}")
            logging.error(traceback.format_exc())
    results_dict = {
        "fl": results_fl,
        "all_data": results_all_data,
        "one_model_per_client": results_one_model_per_client,
    }

    list_of_results_dfs = []

    def parse_dict(d, extract="value"):
        df = pd.DataFrame.from_dict(d)
        df = df.applymap(lambda x: x[extract]).transpose()
        return df

    for _, value in results_dict.items():
        list_of_results_dfs.append(parse_dict(value))
