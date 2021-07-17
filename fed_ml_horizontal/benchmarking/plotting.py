import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metrics_hist(history, plotname_prefix, output_dir):
    """Plots defined metrics over epochs.

    Args:
        history (Box): dict-like Box object containing values of defined metrics over epochs
        plotname_prefix (str): plotname prefix depending of the setting
        output_dir (str): individual output path of executed scenario
    """
    # Get number of epochs
    epochs = range(1, len(history["binary_accuracy"]) + 1)

    fig = plt.figure()

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, history["binary_accuracy"])
    plt.plot(epochs, history["val_binary_accuracy"])
    plt.title("Training and validation accuracy")
    fig.savefig(os.path.join(output_dir, f"{plotname_prefix}_acc.png"))

    fig = plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, history["loss"])
    plt.plot(epochs, history["val_loss"])
    plt.title("Training and validation loss")
    plt.show()
    fig.savefig(os.path.join(output_dir, f"{plotname_prefix}_loss.png"))


def aggregate_and_plot_hists(
    df_run_hists, output_path_for_setting, output_path_for_scenario, prefix
):
    """Plots defined metrics over epochs aggregated over number of reruns.
    Saves figures in defined paths.

    Args:
        df_run_hists (pandas DataFrame): df containing values of defined metrics for train/val over runs and epochs
        output_path_for_setting (str): individual output path of executed setting
        output_path_for_scenario (str): individual output path of executed scenario where all plots and results are saved
        prefix (str): prefix for setting
    """
    logging.info(str(df_run_hists))
    df_run_hists_agg = (
        df_run_hists.groupby(["metric", "train/val", "epoch"])["value"]
        .agg(["mean", "median", "min", "max", "std"])
        .reset_index()
    )
    df_run_hists_agg.to_csv(
        os.path.join(output_path_for_setting, f"{prefix}_df_run_hists_agg.csv")
    )
    logging.info(str(df_run_hists_agg))

    # TODO: Add quantiles as "confidence intervals"
    for metric in df_run_hists_agg["metric"].unique():
        plt.figure()
        g = sns.lineplot(
            data=df_run_hists_agg[lambda x: x.metric == metric],
            x="epoch",
            y="median",
            hue="train/val",
        ).set_title(f"Training and validation {metric} (median)")
        g.get_figure().savefig(
            os.path.join(output_path_for_scenario, f"{prefix}_{metric}_median.png")
        )
        plt.close()

    for metric in df_run_hists_agg["metric"].unique():
        fig = plt.figure()
        sns.lineplot(
            data=df_run_hists[lambda x: x.metric == metric].drop("run", axis=1),
            x="epoch",
            y="value",
            hue="train/val",
            ci="sd",
        ).set_title(f"Training and validation {metric} (mean, sd ci)")
        fig.savefig(
            os.path.join(output_path_for_scenario, f"{prefix}_{metric}_mean_sd.png")
        )


def save_metrics_in_df(df_run_hists, metrics, mode, run, epoch):
    """Saves vales of metrics for train/val over runs and epochs in pandas DataFrame.

    Args:
        df_run_hists (pandas DataFrame): df containing values of defined metrics for train/val over runs and epochs
        metrics (OrderedDict): dict containing defined metrics and values
        mode (str): current mode (train or val)
        run (int): current run
        epoch (int): current epoch

    Returns:
        pandas DataFrame: df containing values of defined metrics for train/val over runs and epochs
    """
    for metric, value in metrics.items():
        tmp_dict = {
            "metric": metric,
            "train/val": mode,
            "run": run,
            "epoch": epoch,
            "value": value,
        }
        df_run_hists = df_run_hists.append(tmp_dict, ignore_index=True)
    return df_run_hists


def create_empty_run_hist_df():
    """Creates empty pandas DataFrame with certain columns

    Returns:
        pandas DataFrame: empty df with columns
    """
    return pd.DataFrame(columns=["metric", "train/val", "run", "epoch", "value"])


def create_dataset_for_plotting(df_run_hists, history, run):
    """Creates dataset of defined metrics for plotting

    Args:
        df_run_hists (pandas DataFrame): df containing values of defined metrics for train/val over runs and epochs
        history (Box): dict-like Box object containing values of defined metrics over epochs
        run (int): current run

    Returns:
        pandas DataFrame: df containing values of defined metrics for train/val over runs and epochs
    """
    hist_df = pd.DataFrame.from_dict(history)
    for mode in ["train", "val"]:
        for metric in [metric for metric in history.keys() if "val_" not in metric]:
            for epoch in range(1, hist_df.shape[0]):
                prefix = "" if mode == "train" else "val_"
                tmp_dict = {
                    "metric": metric,
                    "train/val": mode,
                    "run": run,
                    "epoch": epoch,
                    "value": hist_df.loc[epoch - 1, f"{prefix}{metric}"],
                }
                df_run_hists = df_run_hists.append(tmp_dict, ignore_index=True)

    return df_run_hists
