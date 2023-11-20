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
# NAIRU for the US
df_nairu_us = pd.read_csv(path_data + "us_cbo_nairu.csv")
df_nairu_us = df_nairu_us.rename(columns={"DATE": "quarter", "NROU": "nairu"})
df_nairu_us["quarter"] = pd.to_datetime(df_nairu_us["quarter"]).dt.to_period("q")
df_nairu_us["country"] = "united_states"
df_nairu_us["quarter"] = df_nairu_us["quarter"].astype("str")
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_nairu_us, on=["country", "quarter"], how="outer", validate="one_to_one")
# Sort
df = df.sort_values(by=["country", "quarter"])


# %%
# II --- Pre-analysis wrangling
# Trim countries
list_countries_keep = [
    # "australia",
    # "malaysia",
    # "singapore",
    # "thailand",
    # "indonesia",  # no urate data
    # "philippines",  # no urate data
    "united_states",  # problems with BER
    # "united_kingdom",
    # "germany",
    # "france",
    # "italy",
    # "japan",
    # "south_korea",
    # "taiwan",  # not covered country
    # "hong_kong_sar_china_",  # no core inflation
    # "india",  # no urate data
    # "china",  # special case
    # "chile",
    # "mexico",
    # "brazil",
]
df = df[df["country"].isin(list_countries_keep)]
# Compute NAIRU gap
df["nairu_gap"] = df["urate"] - df["nairu"] 
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu", "expcpi"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = ["stir", "ltir", "urate_ceiling", "urate", "nairu", "privdebt", "privdebt_bank"]
for col in cols_levels:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
for col in cols_rate:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Generate lagged terms for interacted variables
df["urate_int_urate_gap"] = df["urate"] * df["urate_gap"]
df["urate_int_nairu_gap"] = df["urate"] * df["nairu_gap"]
# Generate lags
for lag in range(1, 4 + 1):
    for col in cols_pretransformed + cols_levels + cols_rate:
        df[col + "_lag" + str(lag)] = df.groupby("country")[col].shift(lag)
# Trim dates
df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("q")
df = df[(df["quarter"] >= t_start_q) & (df["quarter"] <= t_end_q)]
# Reset index
df = df.reset_index(drop=True)
# Delete country column
del df["country"]

# %%
# II --- Analysis
# %%
# Chart settings
heatmaps_y_fontsize = 12
heatmaps_x_fontsize = 12
heatmaps_title_fontsize = 12
heatmaps_annot_fontsize = 12
list_file_names = []
# %%
# POLS
# Without REER
eqn = "corecpi ~ 1 + nairu_gap + expcpi + corecpi_lag1"
mod_pols, res_pols, params_table_pols, joint_teststats_pols, reg_det_pols = reg_ols(
    df=df, eqn=eqn
)
file_name = path_output + "phillipscurve_nairu_base_usa_params_pols"
list_file_names += [file_name]
chart_title = "OLS: Without REER (With NAIRU Gap; US-Only)"
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
# With REER
eqn = "corecpi ~ 1 + nairu_gap + expcpi + corecpi_lag1 + reer"
(
    mod_pols_reer,
    res_pols_reer,
    params_table_pols_reer,
    joint_teststats_pols_reer,
    reg_det_pols_reer,
) = reg_ols(df=df, eqn=eqn)
file_name = path_output + "phillipscurve_nairu_base_usa_params_pols_reer"
list_file_names += [file_name]
chart_title = "OLS: With REER (With NAIRU Gap; US-Only)"
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

# %%
# Compile all log likelihoods
df_loglik = pd.DataFrame(
    {
        "Model": [
            "OLS: Without REER",
            "OLS: With REER",
        ],
        "AICc": [
            (-2 * res_pols.llf + 2 * res_pols.df_model)
            + (
                (2 * res_pols.df_model * (res_pols.df_model + 1))
                / (res_pols.nobs - res_pols.df_model - 1)
            ),
            (-2 * res_pols_reer.llf + 2 * res_pols_reer.df_model)
            + (
                (2 * res_pols_reer.df_model * (res_pols_reer.df_model + 1))
                / (res_pols_reer.nobs - res_pols_reer.df_model - 1)
            ),
        ],
        "AIC": [
            (-2 * res_pols.llf + 2 * res_pols.df_model),
            (-2 * res_pols_reer.llf + 2 * res_pols_reer.df_model),
        ],
        "Log-Likelihood": [
            res_pols.llf,
            res_pols_reer.llf,
        ]
    }
)
df_loglik = pd.DataFrame(df_loglik.set_index("Model"))
file_name = path_output + "phillipscurve_nairu_base_usa_loglik"
list_file_names += [file_name]
chart_title = "AICs and Log-Likelihood of Estimated Models \n(With NAIRU Gap)"
fig = heatmap(
    input=df_loglik,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=df_loglik.min().min(),
    ub=df_loglik.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# Compile all heat maps
file_name_pdf = path_output + "phillipscurve_nairu_base_usa_params"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_phillipscurve_nairu_base_usa: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
