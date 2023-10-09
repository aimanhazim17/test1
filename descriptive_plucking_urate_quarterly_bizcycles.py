# %%
import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
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
df = pd.read_parquet(path_data + "data_macro_quarterly_urate.parquet")
country_parameters = pd.read_csv(path_dep + "parameters_by_country_quarterly.csv")
dict_country_x_multiplier = dict(
    zip(country_parameters["country"], country_parameters["x_multiplier_choice"])
)
dict_country_tlb = dict(
    zip(country_parameters["country"], country_parameters["tlb"].fillna(""))
)

# %%
# II --- Cleaning
# Trim countries
list_countries_keep = [
    "australia",
    "malaysia",
    "singapore",
    "thailand",
    "indonesia",
    "philippines",
    "united_states",
    "united_kingdom",
    "germany",
    "france",
    "italy",
    "japan",
    "south_korea",
    # "taiwan",
    "hong_kong_sar_china_",
    "india",
    # "china",
    "chile",
    "mexico",
    "brazil",
]
df = df[df["country"].isin(list_countries_keep)]
# Chronological order by country
df = df.sort_values(by=["country", "quarter"], ascending=[True, True])
df = df.reset_index(drop=True)
# Generate change in unemployment rate (SA)
df["urate_diff"] = df["urate"] - df.groupby("country")["urate"].shift(1)
# Drop NA
df = df.dropna(subset="urate_diff")
# Problematic countries (outliers)
# df = df[~(df["country"] == "japan")]

# %%
# III --- Compute peak-troughs paces, and Exp_{t} ->> Con_{t+1} and Con_{t} ->> Exp_{t+1}; REVERSED DIRECTION
# %%
# Key parameters
col_choice = "urate"
cols_to_compute_ceilings = [col_choice]
col_ref = "urate"


# %%
# Define function
def compute_bizcycle_paces(data, entities_label, rows_per_epi):
    # deep copy
    df = data.copy()
    # blank data frame
    df_consol = pd.DataFrame(
        columns=list(df.columns)
        + [
            col_choice + "peak",
            col_choice + "trough",
            col_choice + "epi",
            col_choice + "pace",
        ]
    )
    df_expcon_consol = pd.DataFrame(
        columns=[entities_label, "expansion_pace", "subsequent_contraction_pace"]
    )
    df_conexp_consol = pd.DataFrame(
        columns=[entities_label, "contraction_pace", "subsequent_expansion_pace"]
    )
    # list of countries
    list_entities = list(df[entities_label].unique())
    # episodes by country
    count_entity = 0
    for entity in tqdm(list_entities):
        # subset
        df_sub = df[df[entities_label] == entity].copy()
        df_sub = df_sub.reset_index(drop=True)
        # restrict time
        tlb = dict_country_tlb[entity]
        if tlb == "":
            pass
        else:
            df_sub["quarter"] = pd.to_datetime(df_sub["quarter"]).dt.to_period("q")
            df_sub = df_sub[df_sub["quarter"] >= tlb]
            df_sub["quarter"] = df_sub["quarter"].astype("str")
        # draw out what threshold multiplier to use
        downturn_threshold_multiplier = dict_country_x_multiplier[entity]
        # which country
        print(
            "Now estimating for "
            + entity
            + ", with threshold of X = "
            + str(round(downturn_threshold_multiplier * df_sub[col_choice].std(), 2))
        )
        # identify peaks and troughs
        df_sub = compute_urate_floor(
            data=df_sub,
            levels_labels=cols_to_compute_ceilings,
            ref_level_label=col_ref,
            time_label="quarter",
            downturn_threshold=downturn_threshold_multiplier,
            bounds_timing_shift=-1,
            hard_bound=True,
        )
        # restrict that  n rows of each episode
        if rows_per_epi is not None:
            d = (
                d.groupby(col_choice + "_epi")
                .head(rows_per_epi)
                .sort_values(by=col_choice + "_epi")
                .reset_index(drop=False)
            )
        elif rows_per_epi is None:
            pass
        # parse episodes and valid countries
        if df_sub[col_choice + "_epi"].max() == 0:
            pass
        else:
            df_sub = df_sub[[col_choice + "_pace", col_choice + "_epi"]]
            df_sub = df_sub.groupby(col_choice + "_epi").agg("mean")
            df_consol = pd.concat([df_consol, df_sub], axis=0)
            # expcon frame
            expansions = df_sub.iloc[::2].reset_index(drop=True)
            subsequent_contractions = df_sub.iloc[1::2].reset_index(drop=True)
            df_expcon = pd.concat(
                [expansions, subsequent_contractions], axis=1
            ).dropna()
            df_expcon.columns = ["expansion_pace", "subsequent_contraction_pace"]
            df_expcon[entities_label] = entity
            df_expcon_consol = pd.concat([df_expcon_consol, df_expcon], axis=0)
            # conexp frame
            contractions = df_sub.iloc[1::2].reset_index(drop=True)
            subsequent_expansions = df_sub.iloc[2::2].reset_index(drop=True)
            df_conexp = pd.concat(
                [contractions, subsequent_expansions], axis=1
            ).dropna()
            df_conexp.columns = ["contraction_pace", "subsequent_expansion_pace"]
            df_conexp[entities_label] = entity
            df_conexp_consol = pd.concat([df_conexp_consol, df_conexp], axis=0)
        # next
        count_entity += 1
    # output
    return df_consol, df_expcon_consol, df_conexp_consol


df, df_expcon, df_conexp = compute_bizcycle_paces(
    data=df, entities_label="country", rows_per_epi=None
)
# df_expcon_avg = df_expcon.groupby("country").agg("mean")
# df_conexp_avg = df_conexp.groupby("country").agg("mean")


# %%
# IV --- Plot them charts
# %%
# Exp --> Con
fig_expcon = scatterplot(
    data=df_expcon,
    y_col="subsequent_contraction_pace",
    y_col_nice="Contraction Pace",
    x_col="expansion_pace",
    x_col_nice="Preceding Expansion Pace",
    marker_colour="black",
    marker_size=9,
    best_fit_colour="black",
    best_fit_width=3,
    main_title="Unemployment Rate: Contraction Pace vs. Preceding Expansion Pace (QoQ SA)",
)
file_name = path_output + "urate_quarterly_bizcycles_expcon"
fig_expcon.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)
# %%
# Con --> Exp
fig_expcon = scatterplot(
    data=df_conexp,
    y_col="subsequent_expansion_pace",
    y_col_nice="Expansion Pace",
    x_col="contraction_pace",
    x_col_nice="Preceding Contraction Pace",
    marker_colour="crimson",
    marker_size=9,
    best_fit_colour="crimson",
    best_fit_width=3,
    main_title="Unemployment Rate: Expansion Pace vs. Preceding Contraction Pace (QoQ SA)",
)
file_name = path_output + "urate_quarterly_bizcycles_conexp"
fig_expcon.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_plucking_urate_quarterly_bizcycles: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
