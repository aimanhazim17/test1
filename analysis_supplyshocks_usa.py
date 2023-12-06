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
    est_varx,
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
t_start_q = "1947Q1"
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
# Extended WTI
df_wti_us = pd.read_csv(path_data + "us_fred_wti.csv")
df_wti_us = df_wti_us.rename(columns={"DATE": "month", "WTISPLC": "wti"})
df_wti_us["quarter"] = pd.to_datetime(df_wti_us["month"]).dt.to_period("q")
df_wti_us = df_wti_us.groupby("quarter")["wti"].mean().reset_index(drop=False)
df_wti_us["quarter"] = df_wti_us["quarter"].astype("str")
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(
    df_nairu_us, on=["country", "quarter"], how="outer", validate="one_to_one"
)
df = df.merge(df_wti_us, on=["quarter"], how="outer")
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
# Compute max-wti (hamilton ??)
df["_zero"] = 0
col_x_cands = []
for i in range(1, 5):
    df["wti" + str(i)] = df["wti"].shift(i)
    col_x_cands = col_x_cands + ["wti" + str(i)]
df["_x"] = df[col_x_cands].max(axis=1)
df["_z"] = 100 * ((df["wti"] / df["_x"]) - 1)
df["maxoil"] = df[["_zero", "_z"]].max(axis=1)
for i in ["_zero", "_x", "_z"] + col_x_cands:
    del df[i]
# Compute max-brent (hamilton ??)
df["_zero"] = 0
col_x_cands = []
for i in range(1, 5):
    df["brent" + str(i)] = df["brent"].shift(i)
    col_x_cands = col_x_cands + ["brent" + str(i)]
df["_x"] = df[col_x_cands].max(axis=1)
df["_z"] = 100 * ((df["brent"] / df["_x"]) - 1)
df["maxoil_brent"] = df[["_zero", "_z"]].max(axis=1)
for i in ["_zero", "_x", "_z"] + col_x_cands:
    del df[i]
df.loc[df["brent"].isna(), "maxoil_brent"] = np.nan
# Compute NAIRU gap
df["nairu_gap"] = df["urate"] - df["nairu"]
# Transform
cols_pretransformed = [
    "rgdp",
    "m2",
    "cpi",
    "corecpi",
    "maxgepu",
    "expcpi",
    "nairu_gap",
    "maxoil",
    "urate_ceiling",
    "nairu",
    "urate",
]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = ["stir", "ltir", "privdebt", "privdebt_bank"]
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
# POLS (u-rate floor)
eqn = (
    "urate_ceiling ~ 1 + maxoil + maxoil_lag1 + maxoil_lag2 + maxoil_lag3 + maxoil_lag4"
)
(
    mod_pols_uratefloor,
    res_pols_uratefloor,
    params_table_pols_uratefloor,
    joint_teststats_pols_uratefloor,
    reg_det_pols_uratefloor,
) = reg_ols(df=df, eqn=eqn)
file_name = path_output + "supplyshocks_usa_pols_uratefloor"
list_file_names += [file_name]
chart_title = "OLS: Supply Shocks (U-Rate Floor; US Only)"
fig = heatmap(
    input=params_table_pols_uratefloor,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_pols_uratefloor.min().min(),
    ub=params_table_pols_uratefloor.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)


# %%
# POLS (nairu)
eqn = "nairu ~ 1 + maxoil + maxoil_lag1 + maxoil_lag2 + maxoil_lag3 + maxoil_lag4"
(
    mod_pols_nairu,
    res_pols_nairu,
    params_table_pols_nairu,
    joint_teststats_pols_nairu,
    reg_det_pols_nairu,
) = reg_ols(df=df, eqn=eqn)
file_name = path_output + "supplyshocks_usa_pols_nairu"
list_file_names += [file_name]
chart_title = "OLS: Supply Shocks (NAIRU; US Only)"
fig = heatmap(
    input=params_table_pols_nairu,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=params_table_pols_nairu.min().min(),
    ub=params_table_pols_nairu.max().max(),
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
            (-2 * res_pols_uratefloor.llf + 2 * res_pols_uratefloor.df_model)
            + (
                (2 * res_pols_uratefloor.df_model * (res_pols_uratefloor.df_model + 1))
                / (res_pols_uratefloor.nobs - res_pols_uratefloor.df_model - 1)
            ),
            (-2 * res_pols_nairu.llf + 2 * res_pols_nairu.df_model)
            + (
                (2 * res_pols_nairu.df_model * (res_pols_nairu.df_model + 1))
                / (res_pols_nairu.nobs - res_pols_nairu.df_model - 1)
            ),
        ],
        "AIC": [
            (-2 * res_pols_uratefloor.llf + 2 * res_pols_uratefloor.df_model),
            (-2 * res_pols_nairu.llf + 2 * res_pols_nairu.df_model),
        ],
        "Log-Likelihood": [
            res_pols_uratefloor.llf,
            res_pols_nairu.llf,
        ],
    }
)
df_loglik = pd.DataFrame(df_loglik.set_index("Model"))
file_name = path_output + "supplyshocks_usa_pols_loglik"
list_file_names += [file_name]
chart_title = "AICs and Log-Likelihood of Estimated Models \n(Supply Shocks; US Only)"
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
# LP analysis
# cols_rate_add = ["urate_ceiling", "nairu", "stir"]
# for col in cols_rate_add:
#     df[col] = df[col] - df[col].shift(4)
# Transform more variables
for urate_col in ["urate_ceiling", "nairu"]:
    # Setup
    endog_base = [
        "maxoil",
    ]  + [urate_col]  # + ["corecpi", "reer", "expcpi"]
    df_sub = df[endog_base].copy()
    # Estimate
    irf = lp.TimeSeriesLP(
        data=df_sub,
        Y=endog_base,
        response=endog_base,
        horizon=12,
        lags=1,
        newey_lags=1,
        ci_width=0.95,
    )
    file_name = path_output + "supplyshocks_lp" + "_" + urate_col + "_usa"
    irf.to_parquet(file_name + ".parquet")
    # Plot
    fig_irf = lp.IRFPlot(
        irf=irf,
        response=endog_base,
        shock=endog_base,
        n_columns=len(endog_base),
        n_rows=len(endog_base),
        maintitle="IRFs (US)",
        show_fig=False,
        save_pic=False,
        out_path="",
        out_name="",
        annot_size=14,
        font_size=16,
    )
    list_file_names += [file_name]
    fig_irf.write_image(file_name + ".png", height=1080, width=1920)
    telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)


# %%
# Compile all heat maps
file_name_pdf = path_output + "supplyshocks_usa"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_supplyshocks_usa: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
