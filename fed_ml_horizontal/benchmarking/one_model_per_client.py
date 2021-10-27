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
    max_num_epochs,
    learning_rate,
    early_stopping_patience,
    early_stopping_monitor,
):
    """Executes one model per client for defined number of runs.
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

    Returns:
        OrderedDict: dict containing results of all runs for this settings
    """

    output_path_for_setting = os.path.join(output_path_for_scenario, "client_models")
    os.makedirs(output_path_for_setting)

    # loss decreases if getting better
    if early_stopping_monitor == "loss":
        mode = "min"
    # other monitors as auc or accuracy are increasing if getting better
    else:
        mode = "max"

    results_one_model_per_client = {}
    for client in client_dataset_dict.keys():
        output_path_for_client = os.path.join(output_path_for_setting, client)
        os.makedirs(output_path_for_client)

        df_run_hists_client_model = create_empty_run_hist_df()

        for i in range(1, num_reruns + 1):
            logging.info(
                f"Training {client} in 'one model per client' scenario, run number {i} of {num_reruns}"
            )

            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_" + early_stopping_monitor,
                patience=early_stopping_patience,
                mode=mode,
                restore_best_weights=True,
            )

            train, test, valid = per_client_train_test_valid(
                client_dataset_dict,
                client_name=client,
                all_images_path=all_images_path,
                batch_size=20,
            )
            client_model = create_my_model()
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            )
            history = client_model.fit(
                train,
                validation_data=valid,
                batch_size=None,
                epochs=max_num_epochs,
                callbacks=[callback],
            )

            df_run_hists_client_model = create_dataset_for_plotting(
                df_run_hists_client_model, history.history, run=i
            )

            if len(history.history["loss"]) == max_num_epochs:
                best_epoch = max_num_epochs
                logging.info(
                    f"Stopped after maximum number of epochs {best_epoch}, early stopping not triggered"
                )
            else:
                best_epoch = len(history.history["loss"]) - early_stopping_patience
                logging.info(
                    f"Early stopping rule triggered after {len(history.history['loss'])} epochs. Best epoch: {best_epoch}"
                )
            if f"run_{i}" not in results_one_model_per_client.keys():
                results_one_model_per_client[f"run_{i}"] = {}

            # test
            test_auc = client_model.evaluate(test)[2]
            logging.info(f"Test AUC: {test_auc}")
            results_one_model_per_client[f"run_{i}"][client] = {
                "metric": "auc",
                "value": test_auc,
                "best_epoch": best_epoch,
            }

        df_run_hists_client_model.to_csv(
            os.path.join(output_path_for_client, f"{client}_df_run_hists.csv")
        )
        aggregate_and_plot_hists(
            df_run_hists_client_model,
            output_path_for_client,
            prefix=f"{client}",
        )
    return results_one_model_per_client


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
