# %%
# Only for a few fringe countries with data issues
# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import (
    telsendmsg,
    telsendimg,
    telsendfiles,
    get_data_from_ceic,
    x13_deseasonalise,
)
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
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
t_start = date(1947, 1, 1)

# %%
# I --- Load data from CEIC
seriesids_all = pd.read_csv(path_ceic + "ceic_macro_quarterly_urate_nsa" + ".csv")
count_col = 0
for col in list(seriesids_all.columns):
    # subset column by column
    seriesids = seriesids_all[col].dropna()
    seriesids = seriesids.astype("str")
    seriesids = [i.replace(".0", "") for i in seriesids]  # remove trailing decimals
    seriesids = [re.sub("[^0-9]+", "", i) for i in list(seriesids)]  # keep only number
    seriesids = [int(i) for i in seriesids]  # convert into list of int
    # pull from ceic one by one
    print("Now downloading " + col)
    # print(', '.join([str(i) for i in seriesids]))
    df_sub = get_data_from_ceic(
        series_ids=seriesids, start_date=t_start, historical_extension=True
    )
    # wrangle
    df_sub = df_sub.reset_index()
    df_sub = df_sub.rename(columns={"date": "date", "country": "country", "value": col})
    # collapse into monthly
    df_sub["quarter"] = pd.to_datetime(df_sub["date"]).dt.to_period("q")
    df_sub = df_sub.groupby(["quarter", "country"])[col].mean().reset_index(drop=False)
    df_sub = df_sub[["quarter", "country", col]]
    # merge
    if count_col == 0:
        df = df_sub.copy()
    elif count_col > 0:
        df = df.merge(df_sub, on=["quarter", "country"], how="outer")
    # next
    count_col += 1
df = df.reset_index(drop=True)
# save interim copy
df["quarter"] = df["quarter"].astype("str")
df.to_parquet(path_data + "data_macro_quarterly_urate_nsa_raw" + ".parquet")

# %%
# II --- Wrangle
# Read downloaded data
df = pd.read_parquet(path_data + "data_macro_quarterly_urate_nsa_raw" + ".parquet")
# Set groupby cols
cols_groups = ["country", "quarter"]
# Sort
df = df.sort_values(by=cols_groups, ascending=[True, True])
# Drop NAs
df = df.dropna(subset=["urate"], axis=0)
# Reset indices
df = df.reset_index(drop=True)
# Now seasonally adjust the series
df_sa = pd.DataFrame(columns=cols_groups + ["urate"])
for country in list(df["country"].unique()):
    # subset
    df_sub = df[df["country"] == country].copy()
    # set periodindex
    df_sub["quarter"] = pd.to_datetime(df_sub["quarter"]).dt.to_period("q")
    df_sub = df_sub.set_index("quarter")
    # adjust
    df_sub = x13_deseasonalise(
        data=df_sub,
        cols_to_adj=["urate"]
    )
    # wrangle
    df_sub = df_sub.reset_index(drop=False)
    df_sub["quarter"] = df_sub["quarter"].astype("str")
    # merge
    df_sa = pd.concat([df_sa, df_sub], axis=0)

# Save processed output
df_sa.to_parquet(path_data + "data_macro_quarterly_urate_nsa_sadjusted" + ".parquet")

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- compile_data_macro_quarterly_urate_nsa: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
