# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, get_data_from_ceic
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
t_start_q = "2012Q1"
t_end_q = "2023Q1"

# %%
# I --- Load data
# Macro
df = pd.read_parquet(path_data + "data_macro_quarterly.parquet")
# UGap
df_ugap = pd.read_parquet(path_output + "plucking_ugap.parquet")
df_ugap["quarter"] = pd.to_datetime(df_ugap["month"]).dt.to_period("q")
df_ugap = (
    df_ugap.groupby(["country", "quarter"])[["urate_ceiling", "urate_gap"]]
    .mean()
    .reset_index(drop=False)
)
df_ugap["quarter"] = df_ugap["quarter"].astype("str")
# Merge
df = df.merge(df_ugap, on=["country", "quarter"], how="outer", validate="one_to_one")

# %%
# II --- Pre-analysis wrangling
# Trim countries
list_countries_keep = [
    "australia",
    "malaysia",
    "singapore",
    "thailand",
    "indonesia",
    "philippines",
    # "united_states",
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
# Transform
cols_pretransformed = ["rgdp", "m2", "cpi", "corecpi", "maxgepu"]
cols_levels = ["reer", "ber", "brent", "gepu"]
cols_rate = ["stir", "ltir", "urate_ceiling", "urate_gap"]
for col in cols_levels:
    df[col] = 100 * ((df[col] / df.groupby("country")[col].shift(4)) - 1)
for col in cols_rate:
    df[col] = df[col] - df.groupby("country")[col].shift(4)
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
endog_base = ["brent", "stir", "ber", "urate", "corecpi"]
exog_base = ["gepu", "maxgepu"]
# Estimate
irf = lp.PanelLPX(
    data=df,
    Y=endog_base,
    X=exog_base,
    response=endog_base,
    horizon=16,
    lags=2,
    varcov="kernel",
    ci_width=0.95,
)
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
    annot_size=7,
    font_size=8,
)
file_name = path_output + "macrodynamics_ugap_lp_irf"
fig_irf.write_image(file_name + ".png")
telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config, msg="global-plucking --- analysis_macrodynamics_ugap: COMPLETED"
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
