import logging
import os

import nest_asyncio
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_federated as tff
from box import Box

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
    num_epochs,
):
    """Executes federated model for defined number of runs.
    Calculates performance metrics for each epoch and generates and saves results and plots.

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        num_reruns (int): number of reruns specified in config object
    """
    # Data is the same for all reruns of the federated model
    fl_train_list, fl_test_list, fl_valid_list = create_fl_datasets(
        client_dataset_dict, all_images_path
    )
    output_path_for_setting = os.path.join(output_path_for_scenario, "fl")
    os.makedirs(output_path_for_setting)

    df_run_hists = create_empty_run_hist_df()
    per_client_df_run_hists = {
        client: create_empty_run_hist_df() for client in client_dataset_dict.keys()
    }

    for i in range(1, num_reruns + 1):
        logging.info(f"Start run {i} out of {num_reruns} runs")
        # TODO: Move to federated_learning.py challenge: input_spec has to be handed over to model_fn
        # Wrap a Keras model for use with TFF.
        def model_fn():
            """Runs tensorflow federated model

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
                    # tfa.metrics.F1Score(num_classes=2),
                ],
            )

        # Simulate a few rounds of training with the selected client devices.
        trainer = tff.learning.build_federated_averaging_process(
            model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.Adam()
        )

        nest_asyncio.apply()  # ? quick and dirty
        state = trainer.initialize()

        fed_hist = Box(
            {
                "binary_accuracy": [],
                "loss": [],
                "val_binary_accuracy": [],
                "val_loss": [],
            }
        )

        federated_eval = tff.learning.build_federated_evaluation(model_fn)
        for epoch in range(1, num_epochs + 1):
            state, metrics = trainer.next(state, fl_train_list)
            fed_hist["binary_accuracy"].append(
                float(metrics["train"]["binary_accuracy"])
            )
            fed_hist["loss"].append(float(metrics["train"]["loss"]))

            logging.info(
                f"epoch {epoch}: train: binary accuracy: {float(metrics['train']['binary_accuracy'])}, auc: {float(metrics['train']['auc'])}, loss: {float(metrics['train']['loss'])}"
            )

            eval_metrics = federated_eval(state.model, fl_test_list)
            fed_hist["val_binary_accuracy"].append(
                float(eval_metrics["binary_accuracy"])
            )
            fed_hist["val_loss"].append(float(eval_metrics["loss"]))
            logging.info(
                f"epoch {epoch}: val: binary accuracy: {float(eval_metrics['binary_accuracy'])}, auc: {float(eval_metrics['auc'])}, loss: {float(eval_metrics['loss'])}"
            )

            df_run_hists = save_metrics_in_df(
                df_run_hists, metrics["train"], mode="train", run=i, epoch=epoch
            )
            df_run_hists = save_metrics_in_df(
                df_run_hists, eval_metrics, mode="val", run=i, epoch=epoch
            )

            # save run hists per client on specific client dataset
            for client_num, client in enumerate(per_client_df_run_hists.keys()):
                metrics_train = evaluate_model_for_client_dataset(
                    state, fl_train_list[client_num]
                )
                logging.info(f"epoch {epoch}: {client}: train: {metrics_train}")
                per_client_df_run_hists[client] = save_metrics_in_df(
                    per_client_df_run_hists[client],
                    metrics_train,
                    mode="train",
                    run=i,
                    epoch=epoch,
                )
                metrics_val = evaluate_model_for_client_dataset(
                    state, fl_test_list[client_num]
                )
                logging.info(f"epoch {epoch}: {client}: val: {metrics_val}")
                per_client_df_run_hists[client] = save_metrics_in_df(
                    per_client_df_run_hists[client],
                    metrics_val,
                    mode="val",
                    run=i,
                    epoch=epoch,
                )

    df_run_hists.to_csv(os.path.join(output_path_for_setting, "fl_df_run_hists.csv"))

    aggregate_and_plot_hists(df_run_hists, output_path_for_setting, prefix="fl")
    for client in per_client_df_run_hists.keys():
        per_client_df_run_hists[client].to_csv(
            os.path.join(output_path_for_setting, f"fl_{client}_df_run_hists.csv")
        )
        aggregate_and_plot_hists(
            per_client_df_run_hists[client],
            output_path_for_setting,
            prefix=f"fl_{client}",
        )


def create_fl_datasets(client_dataset_dict, all_images_path, repeat=1, batch=20):
    """Creates train, test and valid datasets for federated learning

    Args:
        client_dataset_dict (dict): dictionary containing filenames of selected total, train, test, valid images (pitting and no_pitting) for each client
        all_images_path (str): path to saved images
        repeat (int, optional): TODO what's that exactly?. Defaults to 1.
        batch (int, optional): batch size. Defaults to 20.

    Returns:
        list: lists of train, test and valid datasets of all clients
    """
    fl_train_list = []  # list of datasets. one dataset per client
    fl_test_list = []
    fl_valid_list = []

    for client_name, client_data_dict in client_dataset_dict.items():
        fl_train = create_tf_dataset(client_data_dict["train"], all_images_path)
        fl_train_list.append(
            fl_train.shuffle(len(fl_train)).repeat(repeat).batch(batch)
        )
        fl_test = create_tf_dataset(client_data_dict["test"], all_images_path)
        fl_test_list.append(fl_test.shuffle(len(fl_test)).repeat(repeat).batch(batch))
        fl_valid = create_tf_dataset(client_data_dict["valid"], all_images_path)
        fl_valid_list.append(
            fl_valid.shuffle(len(fl_valid)).repeat(repeat).batch(batch)
        )

    return fl_train_list, fl_test_list, fl_valid_list


def evaluate_model_for_client_dataset(state, client_dataset):
    keras_model = create_my_model()
    keras_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(name="auc"),
            # tfa.metrics.F1Score(num_classes=2),
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
