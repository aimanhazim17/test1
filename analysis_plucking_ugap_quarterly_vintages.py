# %%
import pandas as pd
import numpy as np
from helper import telsendfiles, telsendimg, telsendmsg
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

list_t_cutoff = [None, "2019Q4", "2018Q1"]

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
# II --- Additional wrangling
# First difference
df["urate_diff"] = df["urate"] - df.groupby("country")["urate"].shift(1)
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
    "china",
    "chile",
    "mexico",
    "brazil",
]
df = df[df["country"].isin(list_countries_keep)]


# %%
# III --- Compute urate floor
def capybara(df: pd.DataFrame, t_cutoff: str = None):
    # Key parameters
    # downturn_threshold_multiplier = 1  # x times standard deviation
    col_choice = "urate"
    cols_to_compute_ceilings = [col_choice]
    col_ref = "urate"
    # Compute ceilings country-by-country
    count_country = 0
    for country in tqdm(list(df["country"].unique())):
        # restrict country
        if t_cutoff is not None:
            df_sub = df[(df["country"] == country) & (df["quarter"] <= t_cutoff)].copy()
        elif t_cutoff is None:
            df_sub = df[df["country"] == country].copy()
        df_sub = df_sub.reset_index(drop=True)
        # restrict time
        tlb = dict_country_tlb[country]
        if tlb == "":
            pass
        else:
            df_sub["quarter"] = pd.to_datetime(df_sub["quarter"]).dt.to_period("q")
            df_sub = df_sub[df_sub["quarter"] >= tlb]
            df_sub["quarter"] = df_sub["quarter"].astype("str")
        # draw out what threshold multiplier to use
        downturn_threshold_multiplier = dict_country_x_multiplier[country]
        # which country
        print(
            "Now estimating for "
            + country
            + ", with threshold of X = "
            + str(round(downturn_threshold_multiplier * df_sub[col_choice].std(), 2))
        )
        # compute ceiling (same function is fine, as the DNS algo is stepwise when looking backwards)
        df_ceiling_sub = compute_urate_floor(
            data=df_sub,
            levels_labels=cols_to_compute_ceilings,
            ref_level_label=col_ref,
            time_label="quarter",
            downturn_threshold=downturn_threshold_multiplier,
            bounds_timing_shift=-1,
            hard_bound=True,
        )
        # consolidate
        if count_country == 0:
            df_ceiling = df_ceiling_sub.copy()
        elif count_country > 0:
            df_ceiling = pd.concat([df_ceiling, df_ceiling_sub], axis=0)  # top-down
        # next
        count_country += 1
    # Compute urate gap
    df_ceiling["urate_gap"] = df_ceiling["urate"] - df_ceiling["urate_ceiling"]
    # Compute urate gap as ratio (rather than arithmetic distance)
    df_ceiling["urate_gap_ratio"] = df_ceiling["urate"] / df_ceiling["urate_ceiling"]
    # Write down what vintage is this
    if t_cutoff is None:
        df_ceiling["vintage"] = "latest"
    elif t_cutoff is not None:
        df_ceiling["vintage"] = t_cutoff
    # Trim columns
    cols_keep = [
        "vintage",
        "country",
        "quarter",
        "urate_gap",
        "urate_gap_ratio",
        "urate",
        "urate_ceiling",
        "urate_peak",
        "urate_trough",
    ]
    df_ceiling = df_ceiling[cols_keep]
    # Reset indices
    df_ceiling = df_ceiling.reset_index(drop=True)
    # Output
    return df_ceiling


df_ceiling_vintages = pd.DataFrame(
    columns=[
        "vintage",
        "country",
        "quarter",
        "urate_gap",
        "urate_gap_ratio",
        "urate",
        "urate_ceiling",
        "urate_peak",
        "urate_trough",
    ]
)
for t_cutoff in list_t_cutoff:
    df_ceiling = capybara(df=df, t_cutoff=t_cutoff)
    df_ceiling_vintages = pd.concat(
        [df_ceiling_vintages, df_ceiling], axis=0
    )  # top-down


# %%
# IV --- Output
# Local copy
df_ceiling_vintages.to_parquet(path_output + "plucking_ugap_quarterly_vintages.parquet")
df_ceiling_vintages.to_csv(
    path_output + "plucking_ugap_quarterly_vintages.csv", index=False
)


# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_plucking_ugap_quarterly_vintages: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
# Notes:
# Tipping point for MYS is X_mult = 0.3668
