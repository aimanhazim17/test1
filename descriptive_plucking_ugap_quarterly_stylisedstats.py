# %%
import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    subplots_scatterplots,
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
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")

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
# III --- Plot by choice of y-axis
cols_y = ["corecpi", "cpi", "rgdp"]
cols_y_nice = ["Core Inflation", "Inflation", "RGDP Growth"]
plot_colours = ["red", "crimson", "blue"]
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
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_plucking_ugap_quarterly_stylisedstats: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")
