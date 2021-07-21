import logging
import os

import tensorflow as tf
from box import Box

from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.plotting import (
    aggregate_and_plot_hists,
    create_dataset_for_plotting,
    create_empty_run_hist_df,
)
from fed_ml_horizontal.benchmarking.tf_utils import create_tf_dataset


def run_all_data_model(
    client_dataset_dict,
    all_images_path,
    output_path_for_scenario,
    num_reruns,
    num_epochs,
):
    """Executes all data model for defined number of runs.
    Calculates performance metrics for each epoch and generates and saves results and plots.

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        num_reruns (int): number of reruns specified in config object
    """
    (
        all_clients_train,
        all_clients_test,
        all_clients_valid,
    ) = create_ds_for_all_data_model(client_dataset_dict, all_images_path)

    output_path_for_setting = os.path.join(output_path_for_scenario, "all_data")
    os.makedirs(output_path_for_setting)

    df_run_hists_all_data = create_empty_run_hist_df()

    for i in range(1, num_reruns + 1):
        logging.info(f"Start run {i} out of {num_reruns} runs")

        all_data_model = create_my_model()
        all_data_model.compile(
            optimizer="Adam",  # tf.keras.optimizers.Adam(learning_rate=0.0001), #tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        history_all_data = all_data_model.fit(
            all_clients_train,
            validation_data=all_clients_test,
            batch_size=None,
            epochs=num_epochs,
            verbose=1,
        )

        df_run_hists_all_data = create_dataset_for_plotting(
            df_run_hists_all_data, history_all_data.history, run=i
        )
    df_run_hists_all_data.to_csv(
        os.path.join(output_path_for_setting, "all_data_df_run_hists.csv")
    )
    aggregate_and_plot_hists(
        df_run_hists_all_data,
        output_path_for_setting,
        prefix="all_data",
    )


def create_ds_for_all_data_model(client_dataset_dict, all_images_path, batch_size=20):
    """Creates train, test and valid datasets for all data model for all clients

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        batch_size (int, optional): batch size. Defaults to 20.

    Returns:
        BatchDataset: train, test and valid datasets of all clients
    """
    all_clients_train_list = [
        create_tf_dataset(client_dataset_dict[client_name]["train"], all_images_path)
        for client_name in client_dataset_dict.keys()
    ]
    all_clients_train = all_clients_train_list[0]
    for ds in all_clients_train_list[1:]:
        all_clients_train = all_clients_train.concatenate(ds)
    all_clients_train = all_clients_train.shuffle(len(all_clients_train)).batch(
        batch_size
    )

    all_clients_test_list = [
        create_tf_dataset(client_dataset_dict[client_name]["test"], all_images_path)
        for client_name in client_dataset_dict.keys()
    ]
    all_clients_test = all_clients_test_list[0]
    for ds in all_clients_test_list[1:]:
        all_clients_test = all_clients_test.concatenate(ds)
    all_clients_test = all_clients_test.shuffle(len(all_clients_test)).batch(batch_size)

    all_clients_valid_list = [
        create_tf_dataset(client_dataset_dict[client_name]["valid"], all_images_path)
        for client_name in client_dataset_dict.keys()
    ]
    all_clients_valid = all_clients_valid_list[0]
    for ds in all_clients_valid_list[1:]:
        all_clients_valid = all_clients_valid.concatenate(ds)
    all_clients_valid = all_clients_valid.shuffle(len(all_clients_valid)).batch(
        batch_size
    )

    return all_clients_train, all_clients_test, all_clients_valid
