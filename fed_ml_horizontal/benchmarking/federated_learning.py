import gc
import logging
import os

import nest_asyncio
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_federated as tff
from box import Box
from tensorflow_federated.python.core.api import value_base

from fed_ml_horizontal.benchmarking.model import create_my_model
from fed_ml_horizontal.benchmarking.plotting import (
    aggregate_and_plot_hists,
    create_empty_run_hist_df,
    save_metrics_in_df,
)
from fed_ml_horizontal.benchmarking.tf_utils import create_tf_dataset


def run_federated_model(
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
    """Executes federated model for defined number of runs.
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
    # cpu_device = tf.config.list_logical_devices("CPU")[0]
    # tff.backends.native.set_local_execution_context(
    #     server_tf_device=cpu_device, client_tf_devices=[cpu_device]
    # )

    # Data is the same for all reruns of the federated model
    fl_train_list, fl_test_list, fl_valid_list, client_name_list = create_fl_datasets(
        client_dataset_dict, all_images_path
    )
    output_path_for_setting = os.path.join(output_path_for_scenario, "fl")
    os.makedirs(output_path_for_setting)

    df_run_hists = create_empty_run_hist_df()
    per_client_df_run_hists = {
        client: create_empty_run_hist_df() for client in client_name_list
    }

    nest_asyncio.apply()

    results_fl = {}
    for i in range(1, num_reruns + 1):

        results_fl[f"run_{i}"] = {}

        def execute_run(
            i,
            per_client_df_run_hists,
            df_run_hists,
            output_path_for_setting,
            fl_train_list,
            fl_test_list,
            fl_valid_list,
            client_name_list,
            unified_test_dataset,
        ):
            """executes federated learning runs

            Args:
                i (int): current run
                per_client_df_run_hists (pandas DataFrame): df containing values of defined metrics per client for train/val over runs and epochs
                df_run_hists (pandas DataFrame): df containing values of defined metrics for train/val over runs and epochs
                output_path_for_setting (str): individual output path for setting
                fl_train_list (list): lists of train datasets of all clients
                fl_test_list (list): lists of test datasets of all clients
                fl_valid_list (list): lists of validation datasets of all clients
                client_name_list (list): list of all client names
                unified_test_dataset (bool): if true, one unified test dataset is used for all clients and settings

            Returns:
                pandas DataFrame: df containing values of defined metrics for train/val over runs and epochs
                pandas DataFrame: df containing values of defined metrics per client for train/val over runs and epochs
                OrderedDict: dict containing results of all runs for this settings
            """
            logging.info(f"Start run {i} out of {num_reruns} runs")

            def model_fn():
                """Runs tensorflow federated model.

                Returns:
                    tensorflow_federated.python.learning.model_utils.EnhancedModel: keras model with specified loss and metrics
                """
                model = create_my_model()
                return tff.learning.from_keras_model(
                    model,
                    input_spec=fl_train_list[0].element_spec,
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC(),
                    ],
                )

            # Simulate a few rounds of training with the selected client devices.
            trainer = tff.learning.build_federated_averaging_process(
                model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                ),
                # client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                #     learning_rate=learning_rate
                # ),
            )

            # nest_asyncio.apply()
            state = trainer.initialize()

            federated_eval = tff.learning.build_federated_evaluation(model_fn)
            stop = False
            epoch = 1
            early_stopping_monitor_in_de_creasing_since = 0

            while not stop:
                # train
                state, metrics = trainer.next(state, fl_train_list)
                logging.info(
                    f"epoch {epoch}: train: binary accuracy: {float(metrics['train']['binary_accuracy'])}, auc: {float(metrics['train']['auc'])}, loss: {float(metrics['train']['loss'])}"
                )

                # val
                val_metrics = federated_eval(state.model, fl_valid_list)
                logging.info(
                    f"epoch {epoch}: val: binary accuracy: {float(val_metrics['binary_accuracy'])}, auc: {float(val_metrics['auc'])}, loss: {float(val_metrics['loss'])}"
                )

                # test
                if not unified_test_dataset:
                    test_metrics = federated_eval(state.model, fl_test_list)
                else:
                    test_metrics = federated_eval(state.model, [fl_test_list[0]])
                logging.info(
                    f"epoch {epoch}: test: binary accuracy: {float(test_metrics['binary_accuracy'])}, auc: {float(test_metrics['auc'])}, loss: {float(test_metrics['loss'])}"
                )

                # save metrics in df_run_hists dataframe
                df_run_hists = save_metrics_in_df(
                    df_run_hists, metrics["train"], mode="train", run=i, epoch=epoch
                )
                df_run_hists = save_metrics_in_df(
                    df_run_hists, val_metrics, mode="val", run=i, epoch=epoch
                )
                df_run_hists = save_metrics_in_df(
                    df_run_hists, test_metrics, mode="test", run=i, epoch=epoch
                )

                current_early_stopping_monitor = val_metrics[early_stopping_monitor]

                # set initial values in epoch 1
                if epoch == 1:
                    # loss decreases if getting better
                    if early_stopping_monitor == "loss":
                        previous_early_stopping_monitor = (
                            current_early_stopping_monitor + 100
                        )
                    # other monitors as auc or accuracy are increasing if getting better
                    else:
                        previous_early_stopping_monitor = (
                            current_early_stopping_monitor - 100
                        )

                # loss decreases if getting better
                if early_stopping_monitor == "loss":
                    if previous_early_stopping_monitor < current_early_stopping_monitor:
                        early_stopping_monitor_in_de_creasing_since += 1
                    else:
                        early_stopping_monitor_in_de_creasing_since = 0
                # other monitors as auc or accuracy are increasing if getting better
                else:
                    if previous_early_stopping_monitor > current_early_stopping_monitor:
                        early_stopping_monitor_in_de_creasing_since += 1
                    else:
                        early_stopping_monitor_in_de_creasing_since = 0

                # stop if monitor is increasing/decreasing since early_stopping_patience epochs or max_num_epochs is reached
                if (
                    early_stopping_monitor_in_de_creasing_since
                    == early_stopping_patience
                ):
                    stop = True

                    best_epoch = epoch - early_stopping_patience

                    logging.info(
                        f"Early stopping rule triggered after {epoch} epochs. Best epoch: {epoch - early_stopping_patience}"
                    )

                elif epoch == max_num_epochs:
                    stop = True

                    best_epoch = epoch

                    logging.info(
                        f"Stopped after maximum number of epochs {epoch}, early stopping not triggered"
                    )

                previous_early_stopping_monitor = current_early_stopping_monitor

                if stop:
                    best_auc = df_run_hists[
                        lambda x: (x["train/val"] == "test")
                        & (x.epoch == best_epoch)
                        & (x.run == i)
                        & (x.metric == "auc")
                    ].value.values[0]

                    results_fl[f"run_{i}"]["overall"] = {
                        "metric": "auc",
                        "value": best_auc,
                        "best_epoch": best_epoch,
                    }
                    logging.info(f"Test AUC: {best_auc}")

                # save run hists per client on specific client dataset
                for client_num, client in enumerate(client_name_list):
                    # val
                    metrics_val = evaluate_model_for_client_dataset(
                        state, fl_valid_list[client_num]
                    )
                    logging.info(f"epoch {epoch}: {client}: val: {metrics_val}")
                    per_client_df_run_hists[client] = save_metrics_in_df(
                        per_client_df_run_hists[client],
                        metrics_val,
                        mode="val",
                        run=i,
                        epoch=epoch,
                    )
                    # test
                    metrics_test = evaluate_model_for_client_dataset(
                        state, fl_test_list[client_num]
                    )
                    logging.info(f"epoch {epoch}: {client}: test: {metrics_test}")
                    per_client_df_run_hists[client] = save_metrics_in_df(
                        per_client_df_run_hists[client],
                        metrics_test,
                        mode="test",
                        run=i,
                        epoch=epoch,
                    )

                    if stop:
                        best_auc = per_client_df_run_hists[client][
                            lambda x: (x["train/val"] == "test")
                            & (x.epoch == best_epoch)
                            & (x.run == i)
                            & (x.metric == "auc")
                        ].value.values[0]
                        logging.info(f"Test AUC for {client}: {best_auc}")

                        results_fl[f"run_{i}"][client] = {
                            "metric": "auc",
                            "value": best_auc,
                            "best_epoch": best_epoch,
                        }

                epoch = epoch + 1

            del trainer
            del state
            del federated_eval

            return df_run_hists, per_client_df_run_hists, results_fl

        df_run_hists, per_client_df_run_hists, results_fl = execute_run(
            i,
            per_client_df_run_hists,
            df_run_hists,
            output_path_for_setting,
            fl_train_list,
            fl_test_list,
            fl_valid_list,
            client_name_list,
            unified_test_dataset,
        )

        del execute_run
        tf.keras.backend.clear_session()
        gc.collect()

    df_run_hists.to_csv(os.path.join(output_path_for_setting, "fl_df_run_hists.csv"))

    aggregate_and_plot_hists(df_run_hists, output_path_for_setting, prefix="fl")

    for client in client_name_list:
        output_path_for_client = os.path.join(output_path_for_setting, client)
        os.makedirs(output_path_for_client)
        per_client_df_run_hists[client].to_csv(
            os.path.join(output_path_for_client, f"fl_{client}_df_run_hists.csv")
        )
        aggregate_and_plot_hists(
            per_client_df_run_hists[client],
            output_path_for_client,
            prefix=f"fl_{client}",
        )
    return results_fl


def create_fl_datasets(client_dataset_dict, all_images_path, repeat=1, batch=20):
    """Creates train, test and valid datasets for federated learning

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        repeat (int, optional): repeat for batch creation. Defaults to 1.
        batch (int, optional): batch size. Defaults to 20.

    Returns:
        list: lists of train, test and valid datasets of all clients, list of all client names
    """
    fl_train_list = []  # list of datasets. one dataset per client
    fl_test_list = []
    fl_valid_list = []
    client_name_list = []

    for client_name, client_data_dict in client_dataset_dict.items():
        fl_train = create_tf_dataset(client_data_dict["train"], all_images_path)
        fl_train_list.append(
            fl_train.shuffle(len(fl_train) * 10).repeat(repeat).batch(batch)
        )
        fl_test = create_tf_dataset(client_data_dict["test"], all_images_path)
        fl_test_list.append(
            fl_test.shuffle(len(fl_test) * 10).repeat(repeat).batch(batch)
        )
        fl_valid = create_tf_dataset(client_data_dict["valid"], all_images_path)
        fl_valid_list.append(
            fl_valid.shuffle(len(fl_valid) * 10).repeat(repeat).batch(batch)
        )
        client_name_list.append(client_name)

    return fl_train_list, fl_test_list, fl_valid_list, client_name_list


def evaluate_model_for_client_dataset(state, client_dataset):
    """Evaluates federated learning model on individual client datasets.

    Args:
        state (tensorflow_federated.python.learning.framework.optimizer_utils.ServerState): state of the federated learning model in current epoch
        client_dataset (BatchDataset): dataset containing images and labels per client

    Returns:
        dict: dictionary of metrics per client
    """
    keras_model = create_my_model()
    keras_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[  # DONT CHANGE ORDER OF METRICS! BinAcc first, AUC second! Other metrics can be appended, but not prepended!
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    keras_model.set_weights(state.model.trainable)
    metrics_list = keras_model.evaluate(client_dataset)
    metrics = {
        "loss": metrics_list[0],
        "binary_accuracy": metrics_list[1],
        "auc": metrics_list[2],
    }
    return metrics
