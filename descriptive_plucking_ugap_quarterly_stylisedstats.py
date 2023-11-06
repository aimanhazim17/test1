# %%
import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    subplots_scatterplots,
    scatterplot,
    pil_img2pdf,
)
from helper_plucking import compute_urate_floor
from datetime import date, timedelta
import statsmodels.formula.api as smf
import statsmodels.tsa.api as sm
import plotly.graph_objects as go
import plotly.express as px
from ceic_api_client.pyceic import Ceic
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
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")
df = df.merge(df_expcpi, on=["country", "quarter"], how="outer", validate="one_to_one")
# Sort
df = df.sort_values(by=["country", "quarter"])

# %%
# II --- Pre-analysis wrangling
# Trim countries
# list_countries_keep = [
#     "australia",
#     "malaysia",
#     "singapore",
#     "thailand",
#     # "indonesia",  # no urate data
#     # "philippines",  # no urate data
#     # "united_states",  # problems with BER
#     "united_kingdom",
#     "germany",
#     "france",
#     "italy",
#     "japan",
#     "south_korea",
#     # "taiwan",  # not covered country
#     "hong_kong_sar_china_",
#     "india",
#     # "china",  # special case
#     "chile",
#     "mexico",
#     "brazil",
# ]
# df = df[df["country"].isin(list_countries_keep)]
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
        df[col + "_forward" + str(forward)] = df.groupby("country")[col].shift(-1 * forward)
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
# Settings
cols_y = [
    "corecpi",
    "cpi",
    "rgdp",
    "expcpi",
    "corecpi_change",
    "cpi_change",
    "rgdp_change",
    "expcpi_change",
] + [
    "corecpi_forward1",
    "cpi_forward1",
    "rgdp_forward1",
    "expcpi_forward1",
    "corecpi_change_forward1",
    "cpi_change_forward1",
    "rgdp_change_forward1",
    "expcpi_change_forward1",
]
cols_y_nice = [
    "Core Inflation",
    "Inflation",
    "RGDP Growth",
    "Expected Inflation",
    "Change in Core Inflation",
    "Change in Inflation",
    "Change in RGDP Growth",
    "Change in Expected Inflation",
] + [
    "Core Inflation (+1Q)",
    "Inflation (+1Q)",
    "RGDP Growth (+1Q)",
    "Expected Inflation (+1Q)",
    "Change in Core Inflation (+1Q)",
    "Change in Inflation (+1Q)",
    "Change in RGDP Growth (+1Q)",
    "Change in Expected Inflation (+1Q)",
]
plot_colours = [
    "red",
    "crimson",
    "blue",
    "green",
    "orange",
    "peru",
    "cadetblue",
    "darkseagreen",
] * 2
# %%
# Plot by choice of y-axis
for col_y, col_y_nice, plot_colour in zip(cols_y, cols_y_nice, plot_colours):
    list_file_names = []
    for country_groups, snakecase_group_name, nice_group_name, n_rows, n_cols in tqdm(
        zip(
            nested_list_country_groups,
            snakecase_group_names_by_country_groups,
            nice_group_names_by_country_groups,
            rows_by_country_groups,
            cols_by_country_groups,
        )
    ):
        df_sub = df[df["country"].isin(country_groups)].copy()
        fig_urate_and_ceiling = subplots_scatterplots(
            data=df_sub,
            col_group="country",
            cols_x=["urate_gap"],
            cols_y=[col_y],
            annot_size=12,
            font_size=12,
            marker_colours=[plot_colour],
            marker_sizes=[6],
            include_best_fit=True,
            best_fit_colours=[plot_colour],
            best_fit_widths=[2],
            main_title="Quarterly estimated U-rate gap and "
            + col_y_nice
            + " in "
            + nice_group_name,
            maxrows=n_rows,
            maxcols=n_cols,
            add_horizontal_at_yzero=False,
            add_vertical_at_xzero=False,
        )
        file_name = (
            path_output
            + "stylised_stats_plucking_ugap_quarterly_"
            + col_y
            + "_"
            + snakecase_group_name
        )
        fig_urate_and_ceiling.write_image(file_name + ".png")
        # telsendimg(
        #     conf=tel_config,
        #     path=file_name + ".png",
        #     cap=file_name
        # )
        list_file_names += [file_name]
    pdf_file_name = path_output + "stylised_stats_plucking_ugap_quarterly_" + col_y
    pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=pdf_file_name)
    telsendfiles(conf=tel_config, path=pdf_file_name + ".pdf", cap=pdf_file_name)
# %%
# Pool all observations in one chart
list_file_names = []
for col_y, col_y_nice, plot_colour in tqdm(zip(cols_y, cols_y_nice, plot_colours)):
    df_sub = df[[col_y, "urate_gap"]].copy()
    df_sub = df_sub.dropna(axis=0)
    fig = scatterplot(
        data=df_sub,
        y_col=col_y,
        y_col_nice=col_y_nice,
        x_col="urate_gap",
        x_col_nice="U-Rate Gap",
        marker_colour=plot_colour,
        marker_size=6,
        best_fit_colour=plot_colour,
        best_fit_width=2,
        main_title="Quarterly estimated U-rate gap and " + col_y_nice,
    )
    file_name = (
        path_output + "stylised_stats_plucking_ugap_quarterly_" + col_y + "_" + "pooled"
    )
    fig.write_image(file_name + ".png")
    # telsendimg(
    #     conf=tel_config,
    #     path=file_name + ".png",
    #     cap=file_name
    # )
    list_file_names += [file_name]
pdf_file_name = path_output + "stylised_stats_plucking_ugap_quarterly_" + "pooled"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=pdf_file_name)
telsendfiles(conf=tel_config, path=pdf_file_name + ".pdf", cap=pdf_file_name)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_plucking_ugap_quarterly_stylisedstats: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
