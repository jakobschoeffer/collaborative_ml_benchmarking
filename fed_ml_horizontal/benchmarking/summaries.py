import os

import pandas as pd


def calc_welfare_gains(df, ompc_prefix, fl_prefix, output_path):
    """calculates welfare gains per client and overall

    Args:
        df (pandas DataFrame): df containing performance values of current scenario
        ompc_prefix (str): prefix for 'one model per client' setting
        fl_prefix (str): prefix for 'federated learning' setting
        output_path (str): individual output path

    Returns:
        pandas DataFrame: df containing welfare gains per client and overall
    """

    # unnormed welfare gains are commented out

    clients = [x.replace(ompc_prefix, "") for x in df.columns if ompc_prefix in x]
    df_wg = pd.DataFrame(
        columns=clients,
        index=[
            # "WG_AUC_FL",
            "WG_AUC_FL_norm",
            # "WG_AUC_FL_client",
            "WG_AUC_FL_client_norm",
        ],
    )

    for client in clients:
        # # calc WG AUC FL
        # df_wg.loc["WG_AUC_FL", client] = (
        #     df.loc["mean", "FL_overall"] - df.loc["mean", ompc_prefix + client]
        # )
        # calc WG AUC FL norm
        df_wg.loc["WG_AUC_FL_norm", client] = (
            df.loc["mean", "FL_overall"] - df.loc["mean", ompc_prefix + client]
        ) / df.loc["mean", ompc_prefix + client]
        # # calc WG AUC FL client
        # df_wg.loc["WG_AUC_FL_client", client] = (
        #     df.loc["mean", fl_prefix + client] - df.loc["mean", ompc_prefix + client]
        # )
        # calc WG AUC FL client norm
        df_wg.loc["WG_AUC_FL_client_norm", client] = (
            df.loc["mean", fl_prefix + client] - df.loc["mean", ompc_prefix + client]
        ) / df.loc["mean", ompc_prefix + client]

    df_wg.loc[:, "mean"] = df_wg[clients].mean(axis=1)
    df_wg.loc[:, "sum"] = df_wg[clients].sum(axis=1)

    # save df as csv
    df_wg.to_csv(os.path.join(output_path, "welfare_gains.csv"))

    df_wg = pd.read_csv(os.path.join(output_path, "welfare_gains.csv"), index_col=0)
    df_wg = df_wg.transpose()
    df_wg = df_wg * 100
    df_wg.columns = [col + " [%]" for col in df_wg.columns]

    # save as latex file
    with open(os.path.join(output_path, "welfare_gains.tex"), "w") as tf:
        tf.write(
            df_wg.to_latex(
                caption="AUC welfare gains",
                label="tab:auc_welfare",
                position="h",
                float_format="%.2f",
            )
        )

    return df_wg


def extract_performance_results(results_dict, output_path):
    """extracts performance results of all settings and runs from dictionary and calculates mean and standard deviation over all runs

    Args:
        results_dict (OrderedDict): dict containing results of all runs for all settings
        output_path (str): individual output path

    Returns:
        pandas DataFrame: df containing performance values of current scenario
        pandas DataFrame: short version of df containing performance values of current scenario (without detailed run performances)
    """
    list_of_results_dfs = []

    def parse_dict(d, extract="value", prefix=""):
        df = pd.DataFrame.from_dict(d)
        df = df.applymap(lambda x: x[extract]).transpose()
        df = df.add_prefix(prefix)
        return df

    for key, value in results_dict.items():
        list_of_results_dfs.append(parse_dict(d=value, extract="value", prefix=key))

    df_performance = pd.concat(list_of_results_dfs, axis=1)

    df_performance.loc["mean"] = df_performance.mean()
    df_performance.loc["sd"] = df_performance[lambda x: x.index != "mean"].std()

    # save both version as csv and latex file

    # save df as csv
    df_performance.to_csv(os.path.join(output_path, "performance.csv"))

    # save as latex file
    with open(os.path.join(output_path, "performance.tex"), "w") as tf:
        tf.write(
            df_performance.to_latex(
                caption="AUC performance with runs",
                label="tab:auc_performance_runs",
                position="h",
            )
        )

    # save short version of dataframe without detailed run performances
    df_performance_short = df_performance.loc[["mean", "sd"]]
    df_performance_short = df_performance_short.transpose()
    df_performance_short = df_performance_short * 100
    df_performance_short.columns = [
        col + " [%]" for col in df_performance_short.columns
    ]

    # save df as csv
    df_performance_short.to_csv(os.path.join(output_path, "performance_short.csv"))

    # save as latex file
    with open(os.path.join(output_path, "performance_short.tex"), "w") as tf:
        tf.write(
            df_performance_short.to_latex(
                caption="AUC performance",
                label="tab:auc_performance",
                position="h",
                float_format="%.2f",
            )
        )

    return df_performance, df_performance_short
