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
    max_num_epochs,
    learning_rate,
    early_stopping_patience,
    early_stopping_monitor,
    unified_test_dataset,
):
    """Executes all data model for defined number of runs.
    Calculates performance metrics for each epoch and generates and saves results and plots.

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        num_reruns (int): number of reruns specified in config object
        max_num_epochs (int): maximum number of epochs specified in config object
        learning_rate (float): learning rate for optimizer specified in config object
        early_stopping_patience (int): number of epochs with no improvement after which training will be stopped specified in config object
        early_stopping_monitor (str): quantity to be monitored specified in config object
        unified_test_dataset (bool): if true, one unified test dataset is used for all clients and settings
    Returns:
        OrderedDict: dict containing results of all runs for this settings
    """
    (
        all_clients_train,
        all_clients_test,
        all_clients_valid,
    ) = create_ds_for_all_data_model(
        client_dataset_dict, all_images_path, unified_test_dataset
    )

    output_path_for_setting = os.path.join(output_path_for_scenario, "all_data")
    os.makedirs(output_path_for_setting)

    df_run_hists_all_data = create_empty_run_hist_df()

    # loss decreases if getting better
    if early_stopping_monitor == "loss":
        mode = "min"
    # other monitors as auc or accuracy are increasing if getting better
    else:
        mode = "max"

    results_all_data = {}
    for i in range(1, num_reruns + 1):
        logging.info(f"Start run {i} out of {num_reruns} runs")

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_" + early_stopping_monitor,
            patience=early_stopping_patience,
            mode=mode,
            restore_best_weights=True,
        )

        all_data_model = create_my_model()
        all_data_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # "Adam"
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[  # DONT CHANGE ORDER OF METRICS! BinAcc first, AUC second! Other metrics can be appended, but not prepended!
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        # train and val
        history_all_data = all_data_model.fit(
            all_clients_train,
            validation_data=all_clients_valid,
            batch_size=None,
            epochs=max_num_epochs,
            verbose=1,
            callbacks=[callback],
        )

        df_run_hists_all_data = create_dataset_for_plotting(
            df_run_hists_all_data, history_all_data.history, run=i
        )

        if len(history_all_data.history["loss"]) == max_num_epochs:
            best_epoch = max_num_epochs
            logging.info(
                f"Stopped after maximum number of epochs {best_epoch}, early stopping not triggered"
            )
        else:
            best_epoch = len(history_all_data.history["loss"]) - early_stopping_patience
            logging.info(
                f"Early stopping rule triggered after {len(history_all_data.history['loss'])} epochs. Best epoch: {best_epoch}"
            )

        # test
        test_auc = all_data_model.evaluate(all_clients_test)[2]
        logging.info(f"Test AUC: {test_auc}")
        results_all_data[f"run_{i}"] = {}
        results_all_data[f"run_{i}"]["overall"] = {
            "metric": "auc",
            "value": test_auc,
            "best_epoch": best_epoch,
        }

    df_run_hists_all_data.to_csv(
        os.path.join(output_path_for_setting, "all_data_df_run_hists.csv")
    )
    aggregate_and_plot_hists(
        df_run_hists_all_data,
        output_path_for_setting,
        prefix="all_data",
    )
    return results_all_data


def create_ds_for_all_data_model(
    client_dataset_dict, all_images_path, unified_test_dataset, batch_size=20
):
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

    if not unified_test_dataset:
        all_clients_test_list = [
            create_tf_dataset(client_dataset_dict[client_name]["test"], all_images_path)
            for client_name in client_dataset_dict.keys()
        ]
        all_clients_test = all_clients_test_list[0]
        for ds in all_clients_test_list[1:]:
            all_clients_test = all_clients_test.concatenate(ds)
        all_clients_test = all_clients_test.shuffle(len(all_clients_test)).batch(
            batch_size
        )
    else:
        all_clients_test = create_tf_dataset(
            client_dataset_dict[list(client_dataset_dict.keys())[0]]["test"],
            all_images_path,
        )
        all_clients_test = all_clients_test.shuffle(len(all_clients_test)).batch(
            batch_size
        )

    return all_clients_train, all_clients_test, all_clients_valid
