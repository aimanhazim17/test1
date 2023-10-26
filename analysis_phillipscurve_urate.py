# %%
import pandas as pd
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
    "indonesia",  # no urate data
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
# Compute indicator for when urate gap is nil
df.loc[df["urate_gap"] == 0, "urate_gap_is_zero"] = 1
df.loc[df["urate_gap"] > 0, "urate_gap_is_zero"] = 0
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu", "expcpi"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = ["stir", "ltir", "urate_ceiling", "urate", "urate_gap", "urate_gap_ratio", "privdebt", "privdebt_bank"]
for col in cols_levels:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
for col in cols_rate:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Generate lagged terms for interacted variables
df["urate_int_urate_gap_is_zero"] = df["urate"] * df["urate_gap_is_zero"]
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
# Chart settings
heatmaps_y_fontsize = 12
heatmaps_x_fontsize = 12
heatmaps_title_fontsize = 12
heatmaps_annot_fontsize=12
list_file_names = []
# %%
# POLS
# Without REER
eqn = "corecpi ~ 1 + urate * urate_gap_is_zero + expcpi + corecpi_lag1"
mod_pols, res_pols, params_table_pols, joint_teststats_pols, reg_det_pols = reg_ols(
    df=df, eqn=eqn
)
file_name = path_output + "phillipscurve_urate_params_pols"
list_file_names += [file_name]
chart_title = "Pooled OLS: Without REER"
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
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# With REER
eqn = "corecpi ~ 1 + urate * urate_gap_is_zero + expcpi + corecpi_lag1 + reer"
(
    mod_pols_reer,
    res_pols_reer,
    params_table_pols_reer,
    joint_teststats_pols_reer,
    reg_det_pols_reer,
) = reg_ols(df=df, eqn=eqn)
file_name = path_output + "phillipscurve_urate_params_pols_reer"
list_file_names += [file_name]
chart_title = "Pooled OLS: With REER"
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
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# %%
# FE
# Without REER
mod_fe, res_fe, params_table_fe, joint_teststats_fe, reg_det_fe = fe_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
    ],
    i_col="country",
    t_col="time",
    fixed_effects=True,
    time_effects=False,
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_fe"
list_file_names += [file_name]
chart_title = "FE: Without REER"
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
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# With REER
(
    mod_fe_reer,
    res_fe_reer,
    params_table_fe_reer,
    joint_teststats_fe_reer,
    reg_det_fe_reer,
) = fe_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
        "reer",
    ],
    i_col="country",
    t_col="time",
    fixed_effects=True,
    time_effects=False,
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_fe_reer"
list_file_names += [file_name]
chart_title = "FE: With REER"
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
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# GMM-IV
# "n L1.n w k  | gmm(n, 2:4) pred(w k) | onestep nolevel timedumm"
# Without REER
mod_gmmiv, res_gmmiv, params_table_gmmiv = gmmiv_reg(
    df=df,
    eqn="corecpi urate urate_gap_is_zero urate_int_urate_gap_is_zero expcpi L1.corecpi | "
    + "endo(corecpi) pred(urate urate_gap_is_zero urate_int_urate_gap_is_zero expcpi) | " 
    + "hqic collapse",
    i_col="country",
    t_col="time",
)
file_name = path_output + "phillipscurve_urate_params_gmmiv"
list_file_names += [file_name]
chart_title = "GMM-IV: Without REER"
fig = heatmap(
    input=params_table_gmmiv,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_gmmiv.min().min(),
    ub=params_table_gmmiv.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# With REER
mod_gmmiv_reer, res_gmmiv_reer, params_table_gmmiv_reer = gmmiv_reg(
    df=df,
    eqn="corecpi urate urate_gap_is_zero urate_int_urate_gap_is_zero expcpi reer L1.corecpi | "
    + "endo(corecpi) pred(urate urate_gap_is_zero urate_int_urate_gap_is_zero expcpi reer) | " 
    + "hqic collapse",
    i_col="country",
    t_col="time",
)
file_name = path_output + "phillipscurve_urate_params_gmmiv_reer"
list_file_names += [file_name]
chart_title = "GMM-IV: With REER"
fig = heatmap(
    input=params_table_gmmiv_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_gmmiv_reer.min().min(),
    ub=params_table_gmmiv_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# TWFE
# Without REER
mod_twfe, res_twfe, params_table_twfe, joint_teststats_twfe, reg_det_twfe = fe_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
    ],
    i_col="country",
    t_col="time",
    fixed_effects=True,
    time_effects=True,
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_twfe"
list_file_names += [file_name]
chart_title = "TWFE: Without REER"
fig = heatmap(
    input=params_table_twfe,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_twfe.min().min(),
    ub=params_table_twfe.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# With REER
(
    mod_twfe_reer,
    res_twfe_reer,
    params_table_twfe_reer,
    joint_teststats_twfe_reer,
    reg_det_twfe_reer,
) = fe_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
        "reer",
    ],
    i_col="country",
    t_col="time",
    fixed_effects=True,
    time_effects=True,
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_twfe_reer"
list_file_names += [file_name]
chart_title = "TWFE: With REER"
fig = heatmap(
    input=params_table_twfe_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_twfe_reer.min().min(),
    ub=params_table_twfe_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# %%
# RE
# Without REER
mod_re, res_re, params_table_re, joint_teststats_re, reg_det_re = re_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
    ],
    i_col="country",
    t_col="time",
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_re"
list_file_names += [file_name]
chart_title = "RE: Without REER"
fig = heatmap(
    input=params_table_re,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_re.min().min(),
    ub=params_table_re.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# With REER
(
    mod_re_reer,
    res_re_reer,
    params_table_re_reer,
    joint_teststats_re_reer,
    reg_det_re_reer,
) = re_reg(
    df=df,
    y_col="corecpi",
    x_cols=[
        "urate",
        "urate_gap_is_zero",
        "urate_int_urate_gap_is_zero",
        "expcpi",
        "corecpi_lag1",
        "reer",
    ],
    i_col="country",
    t_col="time",
    cov_choice="robust",
)
file_name = path_output + "phillipscurve_urate_params_re_reer"
list_file_names += [file_name]
chart_title = "RE: With REER"
fig = heatmap(
    input=params_table_re_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_re_reer.min().min(),
    ub=params_table_re_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# Compile all heat maps
file_name_pdf = path_output + "phillipscurve_urate_params"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config, msg="global-plucking --- analysis_phillipscurve_urate: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
