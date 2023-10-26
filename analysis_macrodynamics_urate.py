# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, get_data_from_ceic
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
from tabulate import tabulate
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
# Sort
df = df.sort_values(by=["country", "quarter"])


# %%
# II --- Pre-analysis wrangling
# Trim countries
list_countries_keep = [
    "australia",
    "malaysia",  # short urate and corecpi data
    "singapore",
    "thailand",
    "indonesia",  # no ura/te data
    # "philippines",  # no urate data
    "united_states",  # problems with BER
    "united_kingdom",
    "germany",
    "france",
    "italy",
    "japan",
    "south_korea",
    # "taiwan",  # not covered country
    # "hong_kong_sar_china_",  # no core cpi
    "india",  # no urate data
    # "china",  # special case
    "chile",
    "mexico",
    "brazil",
]
df = df[df["country"].isin(list_countries_keep)]
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu", "expcpi"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = [
    "stir",
    "ltir",
    "urate_ceiling",
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
# Check when the panel becomes balanced
min_quarter_by_country = df[
    [
        "country",
        "quarter",
        "expcpi",
        "privdebt",
        "urate_gap_ratio",
        "corecpi",
        "stir",
        "reer",
    ]
].copy()
min_quarter_by_country = min_quarter_by_country.dropna(axis=0)
min_quarter_by_country = (
    min_quarter_by_country.groupby("country")["quarter"].min().reset_index()
)
print(tabulate(min_quarter_by_country, headers="keys", tablefmt="pretty"))
# Trim dates
df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("q")
df = df[(df["quarter"] >= t_start_q) & (df["quarter"] <= t_end_q)]
# Reset index
df = df.reset_index(drop=True)
# Set numeric time index
df["time"] = df.groupby("country").cumcount()
del df["quarter"]
# Set multiindex for localprojections
df = df.set_index(["country", "time"])

# %%
# II --- Analysis
# Setup
endog_base = ["privdebt", "stir", "urate", "corecpi", "reer", "expcpi"]
exog_base = ["brent", "gepu", "maxgepu"]
# Estimate
irf = lp.PanelLPX(
    data=df,
    Y=endog_base,
    X=exog_base,
    response=endog_base,
    horizon=12,
    lags=1,
    varcov="kernel",
    ci_width=0.95,
)
file_name = path_output + "macrodynamics_urate_lp_irf"
irf.to_parquet(file_name + ".parquet")
# Plot
fig_irf = lp.IRFPlot(
    irf=irf,
    response=endog_base,
    shock=endog_base,
    n_columns=len(endog_base),
    n_rows=len(endog_base),
    maintitle="Local Projections Model: Impulse Response Functions",
    show_fig=False,
    save_pic=False,
    out_path="",
    out_name="",
    annot_size=14,
    font_size=16,
)
fig_irf.write_image(file_name + ".png", height=1080, width=1920)
telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config, msg="global-plucking --- analysis_macrodynamics_urate: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
