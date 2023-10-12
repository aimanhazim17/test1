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
list_countries_keep_nice = [
    "Australia",
    "Malaysia",
    "Singapore",
    "Thailand",
    "Indonesia",  # no urate data
    "Philippines",  # no urate data
    "United States",  # problems with BER
    "United Kingdom",
    "Germany",
    "France",
    "Italy",
    "Japan",
    "South Korea",
    # "Taiwan",  # not covered country
    # "Hong Kong",  # no core inflation
    "India",  # no urate data
    # "China",  # special case
    "Chile",
    "Mexico",
    "Brazil",
]
df = df[df["country"].isin(list_countries_keep)]
# Compute indicator for when urate gap is nil
df.loc[df["urate_gap"] == 0, "urate_gap_is_zero"] = 1
df.loc[df["urate_gap"] > 0, "urate_gap_is_zero"] = 0
# Smooth unemployment rate with filter
for country in list_countries_keep:
    cycle, trend = smt.hp_filter.hpfilter(
        df.loc[df["country"] == country, "urate"].dropna(), lamb=1600
    )
    df.loc[df["country"] == country, "urate_smoothed"] = trend
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu", "expcpi"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = [
    "stir",
    "ltir",
    "urate_ceiling",
    "urate_smoothed",
    "urate",
    "urate_gap",
    "urate_gap_ratio",
    "privdebt",
    "privdebt_bank",
]
for col in cols_levels:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
for col in cols_rate:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
# Convert into long-term gaps (unique to this analysis)
cols_convert_to_ltgaps = [
    i
    for i in cols_pretransformed + cols_levels + cols_rate
    if i
    not in ["urate", "urate_smoothed", "urate_ceiling", "urate_gap", "urate_gap_ratio"]
]
for col in cols_convert_to_ltgaps:
    df_ltavg_sub = (
        df.groupby("country")[col]
        .mean()
        .reset_index()
        .rename(columns={col: col + "_ltavg"})
    )
    df = df.merge(df_ltavg_sub, on="country", how="left")
    df[col] = df[col] - df[col + "_ltavg"]
    del df[col + "_ltavg"]
# Generate lagged terms for interacted variables
df["urate_smoothed_int_urate_gap_is_zero"] = df["urate_smoothed"] * df["urate_gap_is_zero"]
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
heatmaps_annot_fontsize = 12
list_file_names = []
# %%
# Country by country loop
relevant_cols = ["urate_smoothed", "urate_smoothed:urate_gap_is_zero"]
params_urate_allcountries = pd.DataFrame(columns=relevant_cols)
params_urate_allcountries_reer = pd.DataFrame(columns=relevant_cols)
for country, country_nice in tqdm(zip(list_countries_keep, list_countries_keep_nice)):
    # Trim country
    df_sub = df[df["country"] == country].copy()
    # OLS
    # Without REER
    eqn = "corecpi ~ 1 + urate_smoothed * urate_gap_is_zero + expcpi + corecpi_lag1 + corecpi_lag2 + corecpi_lag3 + corecpi_lag4"
    mod_ols, res_ols, params_table_ols, joint_teststats_ols, reg_det_ols = reg_ols(
        df=df_sub, eqn=eqn
    )
    params_urate_allcountries = pd.concat(
        [
            params_urate_allcountries,
            pd.DataFrame(params_table_ols.loc[relevant_cols, "Parameter"])
            .transpose()
            .rename(index={"Parameter": country}),
        ],
        axis=0,
    )
    file_name = path_output + "phillipscurve_urate_smoothed_params_ols" + "_" + country
    list_file_names += [file_name]
    chart_title = "OLS: Without REER" + " (" + country_nice + ")"
    fig = heatmap(
        input=params_table_ols,
        mask=False,
        colourmap="vlag",
        outputfile=file_name + ".png",
        title=chart_title,
        lb=params_table_ols.min().min(),
        ub=params_table_ols.max().max(),
        format=".4f",
        show_annot=True,
        y_fontsize=heatmaps_y_fontsize,
        x_fontsize=heatmaps_x_fontsize,
        title_fontsize=heatmaps_title_fontsize,
        annot_fontsize=heatmaps_annot_fontsize,
    )
    # telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
    # With REER
    eqn = "corecpi ~ 1 + urate_smoothed * urate_gap_is_zero + expcpi + corecpi_lag1 + corecpi_lag2 + corecpi_lag3 + corecpi_lag4 + reer"
    (
        mod_ols_reer,
        res_ols_reer,
        params_table_ols_reer,
        joint_teststats_ols_reer,
        reg_det_ols_reer,
    ) = reg_ols(df=df_sub, eqn=eqn)
    params_urate_allcountries_reer = pd.concat(
        [
            params_urate_allcountries_reer,
            pd.DataFrame(params_table_ols_reer.loc[relevant_cols, "Parameter"])
            .transpose()
            .rename(index={"Parameter": country}),
        ],
        axis=0,
    )
    file_name = path_output + "phillipscurve_urate_smoothed_params_ols_reer" + "_" + country
    list_file_names += [file_name]
    chart_title = "OLS: With REER" + " (" + country_nice + ")"
    fig = heatmap(
        input=params_table_ols_reer,
        mask=False,
        colourmap="vlag",
        outputfile=file_name + ".png",
        title=chart_title,
        lb=params_table_ols_reer.min().min(),
        ub=params_table_ols_reer.max().max(),
        format=".4f",
        show_annot=True,
        y_fontsize=heatmaps_y_fontsize,
        x_fontsize=heatmaps_x_fontsize,
        title_fontsize=heatmaps_title_fontsize,
        annot_fontsize=heatmaps_annot_fontsize,
    )
    # telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# Full country heatmaps
file_name = path_output + "phillipscurve_urate_smoothed_cbyc_params_ols"
list_file_names += [file_name]
fig = heatmap(
    input=params_urate_allcountries,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Slope of Estimated Phillips Curve (Smoothed U-Rate) by Countries; Without REER",
    lb=params_urate_allcountries.min().min(),
    ub=params_urate_allcountries.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize - 2,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
file_name = path_output + "phillipscurve_urate_smoothed_cbyc_params_ols_reer"
list_file_names += [file_name]
fig = heatmap(
    input=params_urate_allcountries_reer,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title="Slope of Estimated Phillips Curve (Smoothed U-Rate) by Countries; With REER",
    lb=params_urate_allcountries_reer.min().min(),
    ub=params_urate_allcountries_reer.max().max(),
    format=".4f",
    show_annot=True,
    y_fontsize=heatmaps_y_fontsize,
    x_fontsize=heatmaps_x_fontsize,
    title_fontsize=heatmaps_title_fontsize - 2,
    annot_fontsize=heatmaps_annot_fontsize,
)
# telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# %%
# Compile all heat maps
file_name_pdf = path_output + "phillipscurve_urate_smoothed_cbyc_params"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=file_name_pdf)
telsendfiles(conf=tel_config, path=file_name_pdf + ".pdf", cap=file_name_pdf)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_phillipscurve_urate_smoothed_cbyc: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
