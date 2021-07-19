import logging
import os

import tensorflow as tf
from box import Box

from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.plotting import (
    aggregate_and_plot_hists,
    create_dataset_for_plotting,
    create_empty_run_hist_df,
    plot_metrics_hist,
)
from fed_ml_horizontal.benchmarking.tf_utils import create_tf_dataset


def run_one_model_per_client(
    client_dataset_dict,
    all_images_path,
    output_path_for_scenario,
    num_reruns,
    num_epochs,
):
    """Executes one model per client for defined number of runs.
    Calculates performance metrics for each epoch and generates and saves results and plots.

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        num_reruns (int): number of reruns specified in config object
    """

    output_path_for_setting = os.path.join(output_path_for_scenario, "client_models")
    os.makedirs(output_path_for_setting)

    df_run_hists_client_model = create_empty_run_hist_df()

    for i in range(1, num_reruns + 1):
        output_path_for_run = os.path.join(output_path_for_setting, f"run_{i}")
        os.makedirs(output_path_for_run)

        one_model_per_client_dict = {}
        for client in client_dataset_dict.keys():
            logging.info(
                f"Training {client} in 'one model per client' scenario, run number {i} of {num_reruns}"
            )
            train, test, valid = per_client_train_test_valid(
                client_dataset_dict,
                client_name=client,
                all_images_path=all_images_path,
                batch_size=20,
            )
            client_model = create_my_model()
            client_model.compile(
                optimizer="Adam",  # tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            )
            history = client_model.fit(
                train, validation_data=test, batch_size=None, epochs=num_epochs
            )
            Box(history.history).to_yaml(
                os.path.join(
                    output_path_for_run,
                    f"one_model_per_{client}_train.yaml",
                )
            )
            one_model_per_client_dict.update(
                {
                    client: {
                        "model": client_model,
                        "history": history,
                        "train": train,
                        "test": test,
                        "valid": valid,
                    }
                }
            )

            df_run_hists_client_model = create_dataset_for_plotting(
                df_run_hists_client_model, history.history, run=i
            )

            df_run_hists_client_model.to_csv(
                os.path.join(output_path_for_setting, f"{client}_df_run_hists.csv")
            )
            aggregate_and_plot_hists(
                df_run_hists_client_model,
                output_path_for_setting,
                output_path_for_scenario,
                prefix=f"{client}",
            )

        for client, client_model_dict in one_model_per_client_dict.items():
            logging.info(f"Plotting history for {client}")
            plot_metrics_hist(
                client_model_dict["history"].history,
                f"one_model_per_{client}_plot",
                output_path_for_run,
            )


def per_client_train_test_valid(
    client_dataset_dict, client_name, all_images_path, batch_size=20
):
    """Creates train, test and valid dataset for each client

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        client_name (str): name of client
        all_images_path (str): path to saved images
        batch_size (int, optional): batch size. Defaults to 20.

    Returns:
        BatchDataset: train, test and valid dataset for each client
    """
    client_train = create_tf_dataset(
        client_dataset_dict[client_name]["train"], all_images_path
    )
    client_train = client_train.shuffle(len(client_train)).batch(batch_size)
    client_test = create_tf_dataset(
        client_dataset_dict[client_name]["test"], all_images_path
    )
    client_test = client_test.shuffle(len(client_test)).batch(batch_size)
    client_valid = create_tf_dataset(
        client_dataset_dict[client_name]["valid"], all_images_path
    )
    client_valid = client_valid.shuffle(len(client_valid)).batch(batch_size)

    return client_train, client_test, client_valid
