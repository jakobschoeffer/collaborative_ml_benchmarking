import logging
import os
import time
from contextlib import redirect_stdout

import fed_ml_horizontal.benchmarking.data_splitting as splitting
from fed_ml_horizontal.benchmarking.baseline_model import run_baseline_model
from fed_ml_horizontal.benchmarking.federated_learning import run_federated_model
from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.one_model_per_client import run_one_model_per_client


def run_scenarios(config):
    """Main function that iterates over all scenarios defined in the config file.
    For each scenario the federated, baseline and one model per client setting is performed.

    Args:
        config (Box): config object with project and scenario specifications
    """
    # does not work logging.getLogger("tensorflow").setLevel(logging.ERROR)

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

            client_dataset_dict, all_images_path = splitting.create_client_dataset_dict(
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
            run_federated_model(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                num_epochs=scenario_config.num_epochs,
            )
            logging.info("Finished federated model")

            logging.info("Start baseline model")
            run_baseline_model(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                num_epochs=scenario_config.num_epochs,
            )
            logging.info("Finished baseline model")

            logging.info("Start one model per client")
            run_one_model_per_client(
                client_dataset_dict,
                all_images_path,
                output_path_for_scenario,
                num_reruns=scenario_config.num_reruns,
                num_epochs=scenario_config.num_epochs,
            )
            logging.info("Finished one model per client")
        except:
            logging.error(f"An exception occured trying to run scenario {scenario}")
