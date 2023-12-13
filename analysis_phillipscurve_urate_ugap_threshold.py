# %%
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    reg_ols,
    fe_reg,
    re_reg,
    gmmiv_reg,
    heatmap,
    pil_img2pdf,
)
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
import localprojections as lp
from tabulate import tabulate
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
tel_config = os.getenv("TEL_CONFIG")
t_start_q = "1991Q1"
t_end_q = "2023Q1"

# %%
# I --- Load data
# Macro
df = pd.read_parquet(path_data + "data_macro_quarterly.parquet")
# UGap
df_ugap = pd.read_parquet(path_output + "plucking_ugap_quarterly.parquet")
# df_ugap["quarter"] = pd.to_datetime(df_ugap["month"]).dt.to_period("q")
# df_ugap = (
#     df_ugap.groupby(["country", "quarter"])[["urate_ceiling", "urate_gap"]]
#     .mean()
#     .reset_index(drop=False)
# )
df_ugap["quarter"] = df_ugap["quarter"].astype("str")
# Expected inflation
df_expcpi = pd.read_parquet(path_data + "data_macro_quarterly_expcpi.parquet")
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
# Sort
df = df.sort_values(by=["country", "quarter"])

# %%
# II --- Pre-analysis wrangling
# Trim countries
list_countries_keep = [
    "australia",
    "malaysia",
    "singapore",
    "thailand",
    # "indonesia",  # no urate data
    "philippines",  # no urate data
    "united_states",  # problems with BER
    "united_kingdom",
    "germany",
    "france",
    "italy",
    "japan",
    "south_korea",
    # "taiwan",  # not covered country
    # "hong_kong_sar_china_",  # no core inflation
    "india",  # no urate data
    # "china",  # special case
    "chile",
    "mexico",
    "brazil",
]
df = df[df["country"].isin(list_countries_keep)]
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu", "expcpi"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = ["stir", "ltir", "urate_ceiling", "urate", "privdebt", "privdebt_bank"]
for col in cols_levels:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
for col in cols_rate:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Generate lags
for lag in range(1, 4 + 1):
    for col in cols_pretransformed + cols_levels + cols_rate:
        df[col + "_lag" + str(lag)] = df.groupby("country")[col].shift(lag)
# Trim dates
df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("q")
df = df[(df["quarter"] >= t_start_q) & (df["quarter"] <= t_end_q)]
# Reset index
df = df.reset_index(drop=True)
# Set numeric time index
df["time"] = df.groupby("country").cumcount()
del df["quarter"]


# %%
# II --- Analysis
# %%
# Function
def meowmeowcapybara_pols(
    data: pd.DataFrame,
    y_col: str,
    threshold_input_col: str,
    new_threshold_col_name: str,
    x_interactedwith_threshold_col: str,
    other_x_cols: list[str],
    threshold_range: list[float],
    threshold_range_skip: float,
):
    # deep copy
    df = data.copy()
    # main frame to keep log likelihoods
    df_loglik = pd.DataFrame(columns=["threshold", "loglik"])
    # iterate through threshold candidates
    for threshold in np.arange(
        threshold_range[0], threshold_range[1], threshold_range_skip
    ):
        # create threshold variable
        df.loc[df[threshold_input_col] <= threshold, new_threshold_col_name] = 1
        df.loc[df[threshold_input_col] > threshold, new_threshold_col_name] = 0
        # equation
        eqn = (
            y_col
            + "~"
            + "1"
            + "+"
            + "+".join(other_x_cols)
            + "+"
            + x_interactedwith_threshold_col
            + "*"
            + new_threshold_col_name
        )
        # esitmate
        mod, res, params_table, joint_teststats, reg_det = reg_ols(df=df, eqn=eqn)
        # log likelihood
        df_loglik_sub = pd.DataFrame({"threshold": [threshold], "loglik": [res.llf]})
        df_loglik = pd.concat([df_loglik, df_loglik_sub], axis=0)  # top down
    # find optimal threshold
    threshold_optimal = df_loglik.loc[
        df_loglik["loglik"] == df_loglik["loglik"].max(), "threshold"
    ].reset_index(drop=True)[0]
    # estimate optimal model
    df.loc[df[threshold_input_col] <= threshold_optimal, new_threshold_col_name] = 1
    df.loc[df[threshold_input_col] > threshold_optimal, new_threshold_col_name] = 0
    eqn = (
        y_col
        + "~"
        + "1"
        + "+"
        + "+".join(other_x_cols)
        + "+"
        + x_interactedwith_threshold_col
        + "*"
        + new_threshold_col_name
    )
    mod, res, params_table, joint_teststats, reg_det = reg_ols(df=df, eqn=eqn)
    # output
    return params_table, threshold_optimal, df_loglik


def meowmeowcapybara_fe(
    data: pd.DataFrame,
    y_col: str,
    threshold_input_col: str,
    new_threshold_col_name: str,
    x_interactedwith_threshold_col: str,
    other_x_cols: list[str],
    threshold_range: list[float],
    threshold_range_skip: float,
    entity_col: str,
    time_col: str,
):
    # deep copy
    df = data.copy()
    # main frame to keep log likelihoods
    df_loglik = pd.DataFrame(columns=["threshold", "loglik"])
    # iterate through threshold candidates
    for threshold in np.arange(
        threshold_range[0], threshold_range[1], threshold_range_skip
    ):
        # create threshold variable
        df.loc[df[threshold_input_col] <= threshold, new_threshold_col_name] = 1
        df.loc[df[threshold_input_col] > threshold, new_threshold_col_name] = 0
        # interaction
        df[x_interactedwith_threshold_col + "_" + new_threshold_col_name] = (
            df[x_interactedwith_threshold_col] * df[new_threshold_col_name]
        )
        # estimate
        mod, res, params_table, joint_teststats, reg_det = fe_reg(
            df=df,
            y_col=y_col,
            x_cols=other_x_cols
            + [
                x_interactedwith_threshold_col,
                new_threshold_col_name,
                x_interactedwith_threshold_col + "_" + new_threshold_col_name,
            ],
            i_col=entity_col,
            t_col=time_col,
            fixed_effects=True,
            time_effects=False,
            cov_choice="robust",
        )
        # log likelihood
        df_loglik_sub = pd.DataFrame({"threshold": [threshold], "loglik": [res.loglik]})
        df_loglik = pd.concat([df_loglik, df_loglik_sub], axis=0)  # top down
    # find optimal threshold
    threshold_optimal = df_loglik.loc[
        df_loglik["loglik"] == df_loglik["loglik"].max(), "threshold"
    ].reset_index(drop=True)[0]
    # estimate optimal model
    df.loc[df[threshold_input_col] <= threshold_optimal, new_threshold_col_name] = 1
    df.loc[df[threshold_input_col] > threshold_optimal, new_threshold_col_name] = 0
    df[x_interactedwith_threshold_col + "_int_" + new_threshold_col_name] = (
        df[x_interactedwith_threshold_col] * df[new_threshold_col_name]
    )
    mod, res, params_table, joint_teststats, reg_det = fe_reg(
        df=df,
        y_col=y_col,
        x_cols=other_x_cols
        + [
            x_interactedwith_threshold_col,
            new_threshold_col_name,
            x_interactedwith_threshold_col + "_int_" + new_threshold_col_name,
        ],
        i_col=entity_col,
        t_col=time_col,
        fixed_effects=True,
        time_effects=False,
        cov_choice="robust",
    )
    # output
    return params_table, threshold_optimal, df_loglik


# %%
# Chart settings
heatmaps_y_fontsize = 18
heatmaps_x_fontsize = 12
heatmaps_title_fontsize = 18
heatmaps_annot_fontsize = 18
list_file_names = []
dict_math_greek = {
    "urate": r"$u_{t}$",
    "urate_gap": r"$u^{gap}_{t}$",
    "urate_gap_threshold": r"$1\{u^{gap}_{t} < \tau\}$",
    "urate:urate_gap": r"$u_{t} * u^{gap}_{t}$",
    "urate:urate_gap_threshold": r"$u_{t} * 1\{u^{gap}_{t} < \tau\}$",
    "urate_int_urate_gap": r"$u_{t} * u^{gap}_{t}$",
    "urate_int_urate_gap_threshold": r"$u_{t} * 1\{u^{gap}_{t} < \tau\}$",
    "expcpi": r"$\mathbb{E}_{t}\pi$",
    "corecpi_lag1": r"$\pi_{t-1}$",
    "reer": r"$z_{t}$",
    "Intercept": r"$\alpha$",
}
# %%
# POLS
# Without REER
params_table_pols, threshold_optimal_pols, df_loglik_pols = meowmeowcapybara_pols(
    data=df,
    y_col="corecpi",
    threshold_input_col="urate_gap",
    new_threshold_col_name="urate_gap_threshold",
    x_interactedwith_threshold_col="urate",
    other_x_cols=["expcpi", "corecpi_lag1"],
    threshold_range=[0, df.groupby("country")["urate_gap"].max().min()],
    threshold_range_skip=0.01,
)
file_name = path_output + "phillipscurve_urate_ugap_threshold_params_pols"
list_file_names += [file_name]
chart_title = (
    "Pooled OLS: without REER"
    + "\n (optimal threshold: u-rate gap <= "
    + str(threshold_optimal_pols)
    + ")"
)
params_table_pols = params_table_pols.rename(index=dict_math_greek)
fig = heatmap(
    input=params_table_pols,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_pols.min().min(),
    ub=params_table_pols.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
df_loglik_pols.to_parquet(file_name + "_logliksearch" + ".parquet")
# With REER
(
    params_table_pols_reer,
    threshold_optimal_pols_reer,
    df_loglik_pols_reer,
) = meowmeowcapybara_pols(
    data=df,
    y_col="corecpi",
    threshold_input_col="urate_gap",
    new_threshold_col_name="urate_gap_threshold",
    x_interactedwith_threshold_col="urate",
    other_x_cols=["expcpi", "corecpi_lag1", "reer"],
    threshold_range=[0, df.groupby("country")["urate_gap"].max().min()],
    threshold_range_skip=0.01,
)
file_name = path_output + "phillipscurve_urate_ugap_threshold_params_pols_reer"
list_file_names += [file_name]
chart_title = (
    "Pooled OLS: with REER"
    + "\n (optimal threshold: u-rate gap <= "
    + str(threshold_optimal_pols)
    + ")"
)
params_table_pols_reer = params_table_pols_reer.rename(index=dict_math_greek)
fig = heatmap(
    input=params_table_pols_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_pols_reer.min().min(),
    ub=params_table_pols_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
df_loglik_pols_reer.to_parquet(file_name + "_logliksearch" + ".parquet")
# %%
# FE
# Without REER
params_table_fe, threshold_optimal_fe, df_loglik_fe = meowmeowcapybara_fe(
    data=df,
    y_col="corecpi",
    threshold_input_col="urate_gap",
    new_threshold_col_name="urate_gap_threshold",
    x_interactedwith_threshold_col="urate",
    other_x_cols=["expcpi", "corecpi_lag1"],
    threshold_range=[0, df.groupby("country")["urate_gap"].max().min()],
    threshold_range_skip=0.01,
    entity_col="country",
    time_col="time",
)
file_name = path_output + "phillipscurve_urate_ugap_threshold_params_fe"
list_file_names += [file_name]
chart_title = (
    "FE: without REER"
    + "\n (optimal threshold: u-rate gap <= "
    + str(threshold_optimal_pols)
    + ")"
)
params_table_fe = params_table_fe.rename(index=dict_math_greek)
fig = heatmap(
    input=params_table_fe,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_fe.min().min(),
    ub=params_table_fe.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
df_loglik_fe.to_parquet(file_name + "_logliksearch" + ".parquet")
# With REER
(
    params_table_fe_reer,
    threshold_optimal_fe_reer,
    df_loglik_fe_reer,
) = meowmeowcapybara_fe(
    data=df,
    y_col="corecpi",
    threshold_input_col="urate_gap",
    new_threshold_col_name="urate_gap_threshold",
    x_interactedwith_threshold_col="urate",
    other_x_cols=["expcpi", "corecpi_lag1", "reer"],
    threshold_range=[0, df.groupby("country")["urate_gap"].max().min()],
    threshold_range_skip=0.01,
    entity_col="country",
    time_col="time",
)
file_name = path_output + "phillipscurve_urate_ugap_threshold_params_fe_reer"
list_file_names += [file_name]
chart_title = (
    "FE: with REER"
    + "\n (optimal threshold: u-rate gap <= "
    + str(threshold_optimal_pols)
    + ")"
)
params_table_fe_reer = params_table_fe_reer.rename(index=dict_math_greek)
fig = heatmap(
    input=params_table_fe_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_fe_reer.min().min(),
    ub=params_table_fe_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
df_loglik_fe_reer.to_parquet(file_name + "_logliksearch" + ".parquet")

# %%
# Compute average u-rate gaps
tab_ugap_avg = pd.DataFrame(df.groupby("country")["urate_gap"].mean())
print(tabulate(tab_ugap_avg, headers="keys", tablefmt="pretty"))

tab_ugap_keystats = pd.DataFrame(df["urate_gap"].describe())
print(tabulate(tab_ugap_keystats, headers="keys", tablefmt="pretty"))

df_sub = df[df["urate_gap"] > 0].copy()
tab_ugap_keystats_abovezero = pd.DataFrame(df_sub["urate_gap"].describe())
print(tabulate(tab_ugap_keystats_abovezero, headers="keys", tablefmt="pretty"))

tab_ugap_keystats_panel = pd.DataFrame(df.groupby("country")["urate_gap"].describe())
print(tabulate(tab_ugap_keystats_panel, headers="keys", tablefmt="pretty"))

# %%
# Compile all heat maps
file_name_pdf = path_output + "phillipscurve_urate_ugap_threshold_params"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_phillipscurve_urate_ugap_threshold: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
