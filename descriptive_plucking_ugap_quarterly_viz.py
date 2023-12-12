# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    get_data_from_ceic,
    subplots_linecharts,
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

# %%
# I --- Load data
df = pd.read_parquet(path_output + "plucking_ugap_quarterly.parquet")
# df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("q")
# df["quarter"] = df["quarter"].astype("str")

# %%
# II --- Pre-analysis wrangling
# Trim countries
# list_countries_keep = [
#     "australia",
#     "malaysia",
#     "singapore",
#     "thailand",
#     "indonesia",
#     "philippines",
#     # "united_states",
#     "united_kingdom",
#     "germany",
#     "france",
#     "italy",
#     "japan",
#     "south_korea",
#     # "taiwan",
#     "hong_kong_sar_china_",
#     "india",
#     # "china",
#     "chile",
#     "mexico",
#     "brazil",
# ]
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
# Dictionary to change snake case names to nice names
dict_countries_snake_to_nice = {
    "australia": "Australia",
    "malaysia": "Malaysia",
    "singapore": "Singapore",
    "thailand": "Thailand",
    "indonesia": "Indonesia",
    "philippines": "Philippines",
    "united_states": "United States",
    "united_kingdom": "United Kingdom",
    "germany": "Germany",
    "france": "France",
    "italy": "Italy",
    "japan": "Japan",
    "south_korea": "South Korea",
    "hong_kong_sar_china_": "Hong Kong SAR",
    "india": "India",
    "china": "China",
    "chile": "Chile",
    "mexico": "Mexico",
    "brazil": "Brazil",
}

# %%
# III --- Plot
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
    df_sub["country"] = df_sub["country"].replace(dict_countries_snake_to_nice)
    fig_urate_and_ceiling = subplots_linecharts(
        data=df_sub,
        col_group="country",
        cols_values=["urate", "urate_ceiling"],
        cols_values_nice=["U-Rate", "U-Rate Floor"],
        col_time="quarter",
        annot_size=26,
        font_size=14,
        title_size=28,
        line_colours=["black", "red"],
        line_dashes=["solid", "dash"],
        main_title="Quarterly unemployment rate and estimated floor in "
        + nice_group_name,
        maxrows=n_rows,
        maxcols=n_cols,
    )
    file_name = path_output + "urate_and_ceiling_quarterly_" + snakecase_group_name
    fig_urate_and_ceiling.write_image(file_name + ".png")
    # telsendimg(
    #     conf=tel_config,
    #     path=file_name + ".png",
    #     cap=file_name
    # )
    list_file_names += [file_name]
pdf_file_name = path_output + "urate_and_ceiling_quarterly"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=pdf_file_name)
telsendfiles(conf=tel_config, path=pdf_file_name + ".pdf", cap=pdf_file_name)


# %%
# X --- Notify
telsendmsg(
    conf=tel_config, msg="global-plucking --- descriptive_plucking_ugap_quarterly_viz: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
