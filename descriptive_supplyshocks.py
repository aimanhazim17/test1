# %%
import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    subplots_scatterplots,
    scatterplot,
    subplots_linecharts,
    lineplot,
    lineplot_dualaxes,
    pil_img2pdf,
    heatmap,
)
from helper_plucking import compute_urate_floor
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
import plotly.graph_objects as go
import plotly.express as px
from ceic_api_client.pyceic import Ceic
from tabulate import tabulate
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import ast

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
path_data = "./data/"
path_output = "./output/"
path_ceic = "./ceic/"
path_dep = "./dep/"
tel_config = os.getenv("TEL_CONFIG")

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
# US wages
df_wage_us = pd.read_csv(path_data + "us_fred_wage.csv")
df_wage_us = df_wage_us.rename(columns={"DATE": "month", "AHETPI": "wage"})
df_wage_us["quarter"] = pd.to_datetime(df_wage_us["month"]).dt.to_period("q")
df_wage_us = df_wage_us.groupby("quarter")["wage"].mean().reset_index(drop=False)
df_wage_us["country"] = "united_states"
df_wage_us["quarter"] = df_wage_us["quarter"].astype("str")
# US lfpr
df_lfpr_us = pd.read_csv(path_data + "us_fred_lfpr.csv")
df_lfpr_us = df_lfpr_us.rename(columns={"DATE": "month", "CIVPART": "lfpr"})
df_lfpr_us["quarter"] = pd.to_datetime(df_lfpr_us["month"]).dt.to_period("q")
df_lfpr_us = df_lfpr_us.groupby("quarter")["lfpr"].mean().reset_index(drop=False)
df_lfpr_us["country"] = "united_states"
df_lfpr_us["quarter"] = df_lfpr_us["quarter"].astype("str")
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(
    df_nairu_us, on=["country", "quarter"], how="outer", validate="one_to_one"
)
df = df.merge(df_wti_us, on=["quarter"], how="outer")
df = df.merge(df_wage_us, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_lfpr_us, on=["country", "quarter"], how="outer", validate="one_to_one")
# Sort
df = df.sort_values(by=["country", "quarter"])


# %%
# II --- Pre-analysis wrangling
# Trim countries
countries_asean4 = ["malaysia", "thailand", "indonesia", "philippines"]
countries_asianie = ["singapore", "south_korea", "hong_kong_sar_china_"]
countries_bigemerging = ["china", "india", "mexico", "brazil", "chile"]
countries_adv = [
    "united_states",
    "japan",
    "australia",
    "united_kingdom",
    "germany",
    "france",
    "italy",
]
list_countries_keep = (
    countries_adv + countries_asianie + countries_bigemerging + countries_asean4
)
df = df[df["country"].isin(list_countries_keep)]
# Compute max-wti (hamilton 1996)
df["_zero"] = 0
col_x_cands = []
for i in range(1, 5):
    df["wti" + str(i)] = df.groupby("country")["wti"].shift(i)
    col_x_cands = col_x_cands + ["wti" + str(i)]
df["_x"] = df[col_x_cands].max(axis=1)
df["_z"] = 100 * ((df["wti"] / df["_x"]) - 1)
df["maxoil"] = df[["_zero", "_z"]].max(axis=1)
for i in ["_zero", "_x", "_z"] + col_x_cands:
    del df[i]
df.loc[df["wti"].isna(), "maxoil"] = np.nan
# Compute max-brent (hamilton 1996)
df["_zero"] = 0
col_x_cands = []
for i in range(1, 5):
    df["brent" + str(i)] = df.groupby("country")["brent"].shift(i)
    col_x_cands = col_x_cands + ["brent" + str(i)]
df["_x"] = df[col_x_cands].max(axis=1)
df["_z"] = 100 * ((df["brent"] / df["_x"]) - 1)
df["maxoil_brent"] = df[["_zero", "_z"]].max(axis=1)
for i in ["_zero", "_x", "_z"] + col_x_cands:
    del df[i]
df.loc[df["brent"].isna(), "maxoil_brent"] = np.nan
# Convert to growth
cols_convert_yoy_growth = ["wage"]
for col in cols_convert_yoy_growth:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
# Convert to growth
cols_convert_yoy_diff = ["lfpr"]
for col in cols_convert_yoy_diff:
    df[col + "_yoy"] = df[col] - df.groupby("country")[col].shift(4)
# cols_convert_qoq_diff = ["lfpr"]
# for col in cols_convert_qoq_diff:
#     df[col] = df[col] - df.groupby("country")[col].shift(1)
# Generate change in inflation rates
cols_to_plot = [
    "corecpi",
    "cpi",
    "expcpi",
    "urate",
    "urate_gap",
    "urate_ceiling",
    "rgdp",
    "wage",
    "lfpr",
    "lfpr_yoy",
]
for col in cols_to_plot:
    df[col + "_change"] = df[col] - df.groupby("country")[col].shift(
        1
    )  # first difference in YoY growth rates
# Generate lags
cols_to_plot_change = [i + "_change" for i in cols_to_plot]
cols_to_plot = cols_to_plot + cols_to_plot_change
for forward in range(1, 4 + 1):
    for col in cols_to_plot:
        df[col + "_forward" + str(forward)] = df.groupby("country")[col].shift(
            -1 * forward
        )
# Generate lists for charting
nested_list_country_groups = [
    countries_asean4,
    countries_asianie,
    countries_bigemerging,
    countries_adv,
]
nice_group_names_by_country_groups = ["ASEAN-4", "Asian NIEs", "Major EMs", "AEs"]
snakecase_group_names_by_country_groups = ["asean4", "asianie", "bigemerging", "adv"]
rows_by_country_groups = [2, 2, 2, 3]
cols_by_country_groups = [2, 2, 3, 3]

# %%
# III --- Plot charts
# Set up
df_sub = df[df["country"] == "united_states"].copy()
df_sub = df_sub[
    [
        "quarter",
        "urate",
        "urate_ceiling",
        "nairu",
        "urate_gap",
        "maxoil",
        "maxoil_brent",
        "brent",
        "wti",
        "wage",
        "lfpr",
        "lfpr_yoy",
    ]
]
df_sub["zero"] = 0
# %%
# Plot US u-rate, u-rate floor, and nairu (CBO) + wage inflation + oil shock
cols_sec_y_bools = [False, False, False, False, True]
cols_urates = ["urate", "urate_ceiling", "nairu", "wage", "maxoil"]
cols_urates_nice = ["U-Rate", "U-Rate Floor", "NAIRU", "Wages", "Max-Oil"]
cols_colours = ["black", "crimson", "darkblue", "lightslategrey", "orange"]
cols_width = [2, 2, 2, 2, 2]
cols_dash = ["solid", "dash", "dash", "solid", "solid"]
file_name = path_output + "supplyshocks_maxoil_united_states"
chart_title = (
    "Max-Oil (RHS; Supply Shocks), Wage Inflation, U-Rate, U-Rate Floor, and NAIRU in the US"
)
fig = lineplot_dualaxes(
    data=df_sub,
    y_cols=cols_urates,
    y_cols_nice=cols_urates_nice,
    x_col="quarter",
    x_col_nice="Quarter",
    secondary_y_bools=cols_sec_y_bools,
    primary_y_title="%",
    secondary_y_title="% (Max-Oil)",
    line_colours=cols_colours,
    line_widths=cols_width,
    line_dashes=cols_dash,
    main_title=chart_title,
)
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# Plot US u-rate, u-rate floor, and nairu (CBO) + wage inflation + wti
cols_sec_y_bools = [False, False, False, False, True]
cols_urates = ["urate", "urate_ceiling", "nairu", "wage", "wti"]
cols_urates_nice = ["U-Rate", "U-Rate Floor", "NAIRU", "Wages", "WTI Oil Price"]
cols_colours = ["black", "crimson", "darkblue", "lightslategrey", "orange"]
cols_width = [2, 2, 2, 2, 2]
cols_dash = ["solid", "dash", "dash", "solid", "solid"]
file_name = path_output + "supplyshocks_wti_united_states"
chart_title = (
    "WTI Oil Price (RHS), Wage Inflation, U-Rate, U-Rate Floor, and NAIRU in the US"
)
fig = lineplot_dualaxes(
    data=df_sub,
    y_cols=cols_urates,
    y_cols_nice=cols_urates_nice,
    x_col="quarter",
    x_col_nice="Quarter",
    secondary_y_bools=cols_sec_y_bools,
    primary_y_title="%",
    secondary_y_title="USDpb",
    line_colours=cols_colours,
    line_widths=cols_width,
    line_dashes=cols_dash,
    main_title=chart_title,
)
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)


# %%
# Plot US u-rate, u-rate floor, and nairu (CBO) + wage inflation + lfpr
cols_sec_y_bools = [False, False, False, False, True, True]
cols_urates = ["urate", "urate_ceiling", "nairu", "wage", "lfpr"]
cols_urates_nice = ["U-Rate", "U-Rate Floor", "NAIRU", "Wages", "LFPR"]
cols_colours = ["black", "crimson", "darkblue", "lightslategrey", "darkorange"]
cols_width = [2, 2, 2, 2, 2]
cols_dash = ["solid", "dash", "dash", "solid", "solid"]
file_name = path_output + "supplyshocks_lfpr_united_states"
chart_title = (
    "LFPR (%; RHS), Wage Inflation, U-Rate, U-Rate Floor, and NAIRU in the US"
)
fig = lineplot_dualaxes(
    data=df_sub,
    y_cols=cols_urates,
    y_cols_nice=cols_urates_nice,
    x_col="quarter",
    x_col_nice="Quarter",
    secondary_y_bools=cols_sec_y_bools,
    primary_y_title="%",
    secondary_y_title="% (LFPR)",
    line_colours=cols_colours,
    line_widths=cols_width,
    line_dashes=cols_dash,
    main_title=chart_title,
)
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# Plot US u-rate, u-rate floor, and nairu (CBO) + wage inflation + lfpr + max-oil
cols_sec_y_bools = [False, False, False, False, False, True, False]
cols_urates = ["urate", "urate_ceiling", "nairu", "wage", "lfpr_yoy", "maxoil", "zero"]
cols_urates_nice = ["U-Rate", "U-Rate Floor", "NAIRU", "Wages", "LFPR YoY", "Max-Oil", "Y=0 (LHS)"]
cols_colours = ["black", "crimson", "darkblue", "lightslategrey", "darkgoldenrod", "orange", "black"]
cols_width = [2, 2, 2, 2, 2, 2, 1]
cols_dash = ["solid", "dash", "dash", "solid", "solid", "solid", "solid"]
file_name = path_output + "supplyshocks_maxoil_lfpr_united_states"
chart_title = (
    "LFPR (YoY; LHS), Max-Oil (RHS; Supply Shock), Wage Inflation, U-Rate, U-Rate Floor, and NAIRU in the US"
)
fig = lineplot_dualaxes(
    data=df_sub,
    y_cols=cols_urates,
    y_cols_nice=cols_urates_nice,
    x_col="quarter",
    x_col_nice="Quarter",
    secondary_y_bools=cols_sec_y_bools,
    primary_y_title="%, pp",
    secondary_y_title="% (Max-Oil)",
    line_colours=cols_colours,
    line_widths=cols_width,
    line_dashes=cols_dash,
    main_title=chart_title,
)
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# Tabulate u-rate floor during periods of oil shocks, and otherwise
df_sub.loc[df_sub["maxoil"] == 0, "oil_shock"] = "Max Oil = 0"
df_sub.loc[df_sub["maxoil"] > 0, "oil_shock"] = "Max Oil > 0"
df_sub.loc[df_sub["maxoil"] >= 10, "oil_shock"] = "Max Oil >= 10"
df_sub.loc[df_sub["maxoil"] >= 20, "oil_shock"] = "Max Oil >= 20"
tab_by_oilshocks = df_sub.groupby("oil_shock")[
    ["nairu", "urate_ceiling", "urate", "wage", "lfpr"]
].mean()
chart_title = "Average NAIRU, u-rate floor, u-rate, wage inflation, and LFPR during and outside of periods of oil shocks"
file_name = path_output + "supplyshocks_maxoil_tab_united_states"
fig = heatmap(
    input=tab_by_oilshocks,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=tab_by_oilshocks.min().min(),
    ub=tab_by_oilshocks.max().max(),
    format=".2f",
    show_annot=True,
    y_fontsize=14,
    x_fontsize=14,
    title_fontsize=14,
    annot_fontsize=16,
)
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_supplyshocks: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
