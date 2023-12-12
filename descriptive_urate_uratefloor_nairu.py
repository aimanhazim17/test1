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
    stacked_barchart,
    stacked_barchart_overlaycallouts,
    lineplot,
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
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(
    df_nairu_us, on=["country", "quarter"], how="outer", validate="one_to_one"
)
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
# Generate change in inflation rates
cols_to_plot = [
    "corecpi",
    "cpi",
    "expcpi",
    "urate",
    "urate_gap",
    "urate_ceiling",
    "rgdp",
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
# %%
# Plot US u-rate, u-rate floor, and nairu (CBO)
df_sub = df[df["country"] == "united_states"].copy()
df_sub = df_sub[["quarter", "urate", "urate_ceiling", "nairu", "urate_gap"]]
cols_urates = ["urate", "urate_ceiling", "nairu"]
cols_urates_nice = ["U-Rate", "U-Rate Floor", "NAIRU"]
cols_colours = ["black", "crimson", "darkblue"]
cols_width = [2, 2, 2]
cols_dash = ["solid", "dash", "dash"]
file_name = path_output + "urate_uratefloor_nairu_united_states"
chart_title = "The u-rate, u-rate floor and NAIRU in the US"
fig = lineplot(
    data=df_sub,
    y_cols=cols_urates,
    y_cols_nice=cols_urates_nice,
    x_col="quarter",
    x_col_nice="Quarter",
    line_colours=cols_colours,
    line_widths=cols_width,
    line_dashes=cols_dash,
    main_title=chart_title,
    font_size=24
)
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# %%
# Tabulate how many times the u-rate has crossed the NAIRU, or the u-rate floor
# %%
# heatmap
df_sub.loc[df_sub["urate"] <= df_sub["nairu"], "urate_below_nairu"] = 1
df_sub.loc[df_sub["urate"] > df_sub["nairu"], "urate_below_nairu"] = 0
df_sub.loc[df_sub["urate_gap"] == 0, "urate_at_floor"] = 1
df_sub.loc[df_sub["urate_gap"] > 0, "urate_at_floor"] = 0
tab_urate_nairu_floor = pd.concat(
    [
        df_sub["urate_below_nairu"].value_counts(normalize=True),
        df_sub["urate_at_floor"].value_counts(normalize=True),
    ],
    axis=1,
)
tab_urate_nairu_floor.columns = ["U-Rate at or Below NAIRU", "U-Rate at Floor"]
tab_urate_nairu_floor = 100 * tab_urate_nairu_floor
chart_title = "Percentage of observations in the US stratified by \nwhere u-rate is relative to the NAIRU \nand the floor"
file_name = path_output + "urate_uratefloor_nairu_tabperc_united_states"

fig = heatmap(
    input=tab_urate_nairu_floor,
    mask=False,
    colourmap="vlag",
    outputfile=file_name + ".png",
    title=chart_title,
    lb=tab_urate_nairu_floor.min().min(),
    ub=tab_urate_nairu_floor.max().max(),
    format=".2f",
    show_annot=True,
    y_fontsize=14,
    x_fontsize=14,
    title_fontsize=14,
    annot_fontsize=16,
)
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)
# %%
# stacked bar chart
chart_title = "Percentage of observations in the US stratified by where <br> the u-rate is relative to the NAIRU and the floor"
tab_urate_nairu_floor_flipped = tab_urate_nairu_floor.transpose()
tab_urate_nairu_floor_flipped.index = ["NAIRU", "Plucking u-rate floor"]
tab_urate_nairu_floor_flipped = tab_urate_nairu_floor_flipped.round(2).astype("str")
tab_urate_nairu_floor_flipped.loc[
    "NAIRU", "U-rate at / below NAIRU or at floor (callouts)"
] = ("U-rate at / below NAIRU: <br>" + tab_urate_nairu_floor_flipped.loc["NAIRU", 0] + "%")
tab_urate_nairu_floor_flipped.loc[
    "NAIRU", "U-rate above NAIRU or above floor (callouts)"
] = ("U-rate above NAIRU: <br>" + tab_urate_nairu_floor_flipped.loc["NAIRU", 1] + "%")
tab_urate_nairu_floor_flipped.loc[
    "Plucking u-rate floor", "U-rate at / below NAIRU or at floor (callouts)"
] = (
    "U-rate at floor: <br>"
    + tab_urate_nairu_floor_flipped.loc["Plucking u-rate floor", 0]
    + "%"
)
tab_urate_nairu_floor_flipped.loc[
    "Plucking u-rate floor", "U-rate above NAIRU or above floor (callouts)"
] = (
    "U-rate above floor: <br>"
    + tab_urate_nairu_floor_flipped.loc["Plucking u-rate floor", 1]
    + "%"
)
tab_urate_nairu_floor_flipped = tab_urate_nairu_floor_flipped.reset_index(drop=False)
cols_models = ["Model"]
cols_nairu_floor_comparison = [
    "U-rate at / below NAIRU or at floor",
    "U-rate above NAIRU or above floor",
]
cols_nairu_floor_comparison_callouts = [
    i + " (callouts)" for i in cols_nairu_floor_comparison
]
colours_nairu_floor_comparison = ["lavenderblush", "lightcyan"]
tab_urate_nairu_floor_flipped.columns = (
    cols_models + cols_nairu_floor_comparison + cols_nairu_floor_comparison_callouts
)
tab_urate_nairu_floor_flipped[cols_nairu_floor_comparison] = (
    tab_urate_nairu_floor_flipped[cols_nairu_floor_comparison].astype("float").round(2)
)
fig = stacked_barchart_overlaycallouts(
    data=tab_urate_nairu_floor_flipped,
    stacked_y_cols=cols_nairu_floor_comparison,
    callouts_stacked_y_cols=cols_nairu_floor_comparison_callouts,
    stacked_colours=colours_nairu_floor_comparison,
    x_col=cols_models[0],
    main_title=chart_title,
    decimal_points=2,
    font_size=24,
    bar_callout_size=36,
)
fig.update_layout(showlegend=False)
file_name = path_output + "urate_uratefloor_nairu_tabperc_stackedbar_united_states"
fig.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=chart_title)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_urate_uratefloor_nairu: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
