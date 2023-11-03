# %%
import pandas as pd
import numpy as np
from helper import (
    telsendfiles,
    telsendimg,
    telsendmsg,
    scatterplot,
    pil_img2pdf,
    outlier_isolationforest,
    outlier_tailends,
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
    "china",
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
            col_choice + "_peak",
            col_choice + "_trough",
            col_choice + "_epi",
            col_choice + "_amplitude",
            col_choice + "_first",
            col_choice + "_last",
        ]
    )  # instead of pace, we want amplitude, first, and last
    df_expcon_consol = pd.DataFrame(
        columns=[entities_label, "expansion_amplitude", "subsequent_contraction_amplitude"]
    )
    df_conexp_consol = pd.DataFrame(
        columns=[entities_label, "contraction_amplitude", "subsequent_expansion_amplitude"]
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
            df_sub = (
                df_sub.groupby(col_choice + "_epi")
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
            df_sub = df_sub[
                [col_choice, col_choice + "_epi"]
            ]  # instead of pace, keep the raw input
            # Now generate generate first and last observation per episode
            df_sub_first = (
                df_sub.groupby(col_choice + "_epi")
                .first()
                .rename(columns={col_choice: col_choice + "_first"})
            )
            df_sub_last = (
                df_sub.groupby(col_choice + "_epi")
                .last()
                .rename(columns={col_choice: col_choice + "_last"})
            )
            df_sub = pd.concat([df_sub_first, df_sub_last], axis=1)  # left-right
            # now compute amplitudes
            df_sub[col_choice + "_amplitude"] = df_sub[col_choice + "_last"] - df_sub[col_choice + "_first"] 
            df_sub = pd.DataFrame(df_sub[col_choice + "_amplitude"])
            df_consol = pd.concat([df_consol, df_sub], axis=0)  # top-down
            # expcon frame
            expansions = df_sub.iloc[::2].reset_index(drop=True)
            subsequent_contractions = df_sub.iloc[1::2].reset_index(drop=True)
            df_expcon = pd.concat(
                [expansions, subsequent_contractions], axis=1
            ).dropna()
            df_expcon.columns = ["expansion_amplitude", "subsequent_contraction_amplitude"]
            df_expcon[entities_label] = entity
            df_expcon_consol = pd.concat([df_expcon_consol, df_expcon], axis=0)
            # conexp frame
            contractions = df_sub.iloc[1::2].reset_index(drop=True)
            subsequent_expansions = df_sub.iloc[2::2].reset_index(drop=True)
            df_conexp = pd.concat(
                [contractions, subsequent_expansions], axis=1
            ).dropna()
            df_conexp.columns = ["contraction_amplitude", "subsequent_expansion_amplitude"]
            df_conexp[entities_label] = entity
            df_conexp_consol = pd.concat([df_conexp_consol, df_conexp], axis=0)
        # next
        count_entity += 1
    # output
    return df_consol, df_expcon_consol, df_conexp_consol


df, df_expcon, df_conexp = compute_bizcycle_paces(
    data=df, entities_label="country", rows_per_epi=None
)
df_expcon_avg = df_expcon.groupby("country").agg("mean")
df_conexp_avg = df_conexp.groupby("country").agg("mean")

tailend_threshold = 0.05  # 0.01 0.05 0.1
df_expcon_trimmed = outlier_tailends(
    data=df_expcon,
    cols_x=["subsequent_contraction_amplitude", "expansion_amplitude"],
    perc_trim=tailend_threshold,
)
df_conexp_trimmed = outlier_tailends(
    data=df_conexp,
    cols_x=["subsequent_expansion_amplitude", "contraction_amplitude"],
    perc_trim=tailend_threshold,
)

# df_expcon_trimmed = outlier_isolationforest(
#     data=df_expcon,
#     cols_x=["subsequent_contraction_amplitude", "expansion_amplitude"],
#     opt_max_samples=int(len(df_expcon) / 4),
#     opt_threshold=0.3,
# )
# df_conexp_trimmed = outlier_isolationforest(
#     data=df_conexp,
#     cols_x=["subsequent_expansion_amplitude", "contraction_amplitude"],
#     opt_max_samples=int(len(df_conexp) / 4),
#     opt_threshold=0.3,
# )

# df_expcon_trimmed = df_expcon[
#     (df_expcon["subsequent_contraction_amplitude"] >= 0) & (df_expcon["expansion_amplitude"] <= 0)
# ].copy()
# df_conexp_trimmed = df_conexp[
#     (df_conexp["subsequent_expansion_amplitude"] <= 0) & (df_conexp["contraction_amplitude"] >= 0)
# ].copy()


# %%
# IV --- Plot them charts
list_file_names = []
# %%
# Exp --> Con
fig_expcon = scatterplot(
    data=df_expcon,
    y_col="subsequent_contraction_amplitude",
    y_col_nice="Subsequent Contraction Amplitude",
    x_col="expansion_amplitude",
    x_col_nice="Expansion Amplitude",
    marker_colour="black",
    marker_size=9,
    best_fit_colour="black",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Contraction Amplitude vs. Expansion Amplitude",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_expcon"
list_file_names += [file_name]
fig_expcon.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# Con --> Exp
fig_conexp = scatterplot(
    data=df_conexp,
    y_col="subsequent_expansion_amplitude",
    y_col_nice="Subsequent Expansion Amplitude",
    x_col="contraction_amplitude",
    x_col_nice="Contraction Amplitude",
    marker_colour="crimson",
    marker_size=9,
    best_fit_colour="crimson",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Expansion Amplitude vs. Contraction Amplitude",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_conexp"
list_file_names += [file_name]
fig_conexp.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# Exp --> Con (Trimmed outliers)
fig_expcon_trimmed = scatterplot(
    data=df_expcon_trimmed,
    y_col="subsequent_contraction_amplitude",
    y_col_nice="Subsequent Contraction Amplitude",
    x_col="expansion_amplitude",
    x_col_nice="Expansion Amplitude",
    marker_colour="black",
    marker_size=9,
    best_fit_colour="black",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Contraction Amplitude vs. Expansion Amplitude; Without Outliers",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_expcon_trimmed"
list_file_names += [file_name]
fig_expcon_trimmed.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# Con --> Exp (Trimmed outliers)
fig_conexp_trimmed = scatterplot(
    data=df_conexp_trimmed,
    y_col="subsequent_expansion_amplitude",
    y_col_nice="Subsequent Expansion Amplitude",
    x_col="contraction_amplitude",
    x_col_nice="Contraction Amplitude",
    marker_colour="crimson",
    marker_size=9,
    best_fit_colour="crimson",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Expansion Amplitude vs. Contraction Amplitude; Without Outliers",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_conexp_trimmed"
list_file_names += [file_name]
fig_conexp_trimmed.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# Exp --> Con (Country Avg)
fig_expcon_avg = scatterplot(
    data=df_expcon_avg,
    y_col="subsequent_contraction_amplitude",
    y_col_nice="Subsequent Contraction Amplitude",
    x_col="expansion_amplitude",
    x_col_nice="Expansion Amplitude",
    marker_colour="black",
    marker_size=9,
    best_fit_colour="black",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Contraction Amplitude vs. Expansion Amplitude; Country Averages",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_expcon_avg"
list_file_names += [file_name]
fig_expcon_avg.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# Con --> Exp (Country Avg)
fig_conexp_avg = scatterplot(
    data=df_conexp_avg,
    y_col="subsequent_expansion_amplitude",
    y_col_nice="Subsequent Expansion Amplitude",
    x_col="contraction_amplitude",
    x_col_nice="Contraction Amplitude",
    marker_colour="crimson",
    marker_size=9,
    best_fit_colour="crimson",
    best_fit_width=3,
    main_title="Unemployment Rate: Subsequent Expansion Amplitude vs. Contraction Amplitude; Country Averages",
)
file_name = path_output + "urate_quarterly_bizcycles_amplitude_conexp_avg"
list_file_names += [file_name]
fig_conexp_avg.write_image(file_name + ".png")
# telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# Compile into pdf
pdf_file_name = path_output + "urate_quarterly_bizcycles_amplitude"
pil_img2pdf(list_images=list_file_names, extension="png", pdf_name=pdf_file_name)
telsendfiles(conf=tel_config, path=pdf_file_name + ".pdf", cap=pdf_file_name)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- descriptive_plucking_urate_quarterly_bizcycles_amplitude: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
