import logging
import os

import tensorflow as tf

from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.plotting import (
    aggregate_and_plot_hists,
    create_dataset_for_plotting,
    create_empty_run_hist_df,
)
from fed_ml_horizontal.benchmarking.tf_utils import create_tf_dataset


def run_one_model_per_client(
    client_dataset_dict,
    all_images_path,
    output_path_for_scenario,
    num_reruns,
    num_epochs,
    learning_rate,
):
    """Executes one model per client for defined number of runs.
    Calculates performance metrics for each epoch and generates and saves results and plots.

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        num_reruns (int): number of reruns specified in config object
        num_epochs (int): number of epochs specified in config object
        learning_rate (float): learning rate for optimizer
    """

    output_path_for_setting = os.path.join(output_path_for_scenario, "client_models")
    os.makedirs(output_path_for_setting)

    for client in client_dataset_dict.keys():
        output_path_for_client = os.path.join(output_path_for_setting, client)
        os.makedirs(output_path_for_client)

        df_run_hists_client_model = create_empty_run_hist_df()

        for i in range(1, num_reruns + 1):
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
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                ),  # "Adam"
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            )
            history = client_model.fit(
                train, validation_data=valid, batch_size=None, epochs=num_epochs
            )

            df_run_hists_client_model = create_dataset_for_plotting(
                df_run_hists_client_model, history.history, run=i
            )

        df_run_hists_client_model.to_csv(
            os.path.join(output_path_for_client, f"{client}_df_run_hists.csv")
        )
        aggregate_and_plot_hists(
            df_run_hists_client_model,
            output_path_for_client,
            prefix=f"{client}",
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
