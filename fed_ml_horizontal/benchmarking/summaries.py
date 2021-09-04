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
    clients = [x.replace(ompc_prefix, "") for x in df.columns if ompc_prefix in x]
    df_wg = pd.DataFrame(
        columns=clients,
        index=[
            "WG_AUC_FL",
            "WG_AUC_FL_norm",
            "WG_AUC_FL_client",
            "WG_AUC_FL_client_norm",
        ],
    )

    for client in clients:
        # calc WG AUC FL
        df_wg.loc["WG_AUC_FL", client] = (
            df.loc["mean", "FL_overall"] - df.loc["mean", ompc_prefix + client]
        )
        # calc WG AUC FL norm
        df_wg.loc["WG_AUC_FL_norm", client] = (
            df.loc["mean", "FL_overall"] - df.loc["mean", ompc_prefix + client]
        ) / df.loc["mean", ompc_prefix + client]
        # calc WG AUC FL client
        df_wg.loc["WG_AUC_FL_client", client] = (
            df.loc["mean", fl_prefix + client] - df.loc["mean", ompc_prefix + client]
        )
        # calc WG AUC FL client norm
        df_wg.loc["WG_AUC_FL_client_norm", client] = (
            df.loc["mean", fl_prefix + client] - df.loc["mean", ompc_prefix + client]
        ) / df.loc["mean", ompc_prefix + client]

    df_wg.loc[:, "mean"] = df_wg[clients].mean(axis=1)
    df_wg.loc[:, "sum"] = df_wg[clients].sum(axis=1)

    # save df as csv
    df_wg.to_csv(os.path.join(output_path, "welfare_gains.csv"))

    # save as latex file
    with open(os.path.join(output_path, "performance.tex"), "w") as tf:
        tf.write(df_wg.to_latex())

    return df_wg


def extract_performance_results(results_dict, output_path):
    """extracts performance results of all settings and runs from dictionary and calculates mean and standard deviation over all runs

    Args:
        results_dict (OrderedDict): dict containing results of all runs for all settings
        output_path (str): individual output path

    Returns:
        pandas DataFrame: df containing performance values of current scenario
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

    # save df as csv
    df_performance.to_csv(os.path.join(output_path, "performance.csv"))

    # save as latex file
    with open(os.path.join(output_path, "performance.tex"), "w") as tf:
        tf.write(df_performance.to_latex())

    return df_performance
