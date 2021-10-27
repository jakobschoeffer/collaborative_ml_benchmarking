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
    plt.ylim(bottom=0.7, top=1)
    plt.plot(epochs, history["binary_accuracy"])
    plt.plot(epochs, history["val_binary_accuracy"])
    # plt.title("Training and validation accuracy") # plots without title for paper and thesis
    fig.savefig(os.path.join(output_dir, f"{plotname_prefix}_acc.png"), dpi=600)

    fig = plt.figure()

    # Plot training and validation loss per epoch
    plt.ylim(bottom=0, top=0.5)
    plt.plot(epochs, history["loss"])
    plt.plot(epochs, history["val_loss"])
    # plt.title("Training and validation loss") # plots without title for paper and thesis
    plt.show()
    fig.savefig(os.path.join(output_dir, f"{plotname_prefix}_loss.png"), dpi=600)


def aggregate_and_plot_hists(df_run_hists, output_path, prefix):
    """Plots defined metrics over epochs aggregated over number of reruns.
    Saves figures in defined paths.

    Args:
        df_run_hists (pandas DataFrame): df containing values of defined metrics for train/val over runs and epochs
        output_path (str): individual output path
        prefix (str): prefix for setting
    """
    logging.info(str(df_run_hists))
    df_run_hists_agg = (
        df_run_hists.groupby(["metric", "train/val", "epoch"])["value"]
        .agg(["mean", "median", "min", "max", "std"])
        .reset_index()
    )
    df_run_hists_agg.to_csv(os.path.join(output_path, f"{prefix}_df_run_hists_agg.csv"))
    logging.info(str(df_run_hists_agg))

    palette = {
        "train": sns.color_palette()[0],
        "val": sns.color_palette()[1],
        # "test": sns.color_palette()[2],
    }

    for metric in df_run_hists_agg["metric"].unique():
        plt.figure()
        if metric not in ["loss"]:
            plt.ylim(bottom=0.7, top=1)
        g = sns.lineplot(
            data=df_run_hists_agg[
                lambda x: (x.metric == metric) & (x["train/val"].isin(["train", "val"]))
            ],
            x="epoch",
            y="median",
            hue="train/val",
            palette=palette,
        )  # .set_title(f"Training and validation {metric} (median)") # plots without title for paper and thesis
        g.get_figure().savefig(
            os.path.join(output_path, f"{prefix}_{metric}_median.png"), dpi=600
        )

        plt.close()

    for metric in df_run_hists_agg["metric"].unique():
        fig = plt.figure()
        if metric not in ["loss"]:
            plt.ylim(bottom=0.7, top=1)
        sns.lineplot(
            data=df_run_hists[
                lambda x: (x.metric == metric) & (x["train/val"].isin(["train", "val"]))
            ].drop("run", axis=1),
            x="epoch",
            y="value",
            hue="train/val",
            ci="sd",
            palette=palette,
        )  # .set_title(f"Training and validation {metric} (mean, sd ci)") # plots without title for paper and thesis
        fig.savefig(
            os.path.join(output_path, f"{prefix}_{metric}_mean_sd.png"), dpi=600
        )
        plt.close()


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
            for epoch in range(1, hist_df.shape[0] + 1):
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


def create_boxplots(df, output_path):
    """Creates boxplots of performance dataframe

    Args:
        df (pandas DataFrame): df containing performance values of current scenario
        output_path (str): individual output path
    """
    sns.set_style("whitegrid")
    fig = plt.figure()
    g = sns.boxplot(data=df.loc[lambda x: ["run" in x for x in x.index]])
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    fig = g.get_figure()
    fig.savefig(
        os.path.join(output_path, "performance.png"), bbox_inches="tight", dpi=600
    )
    plt.close()
