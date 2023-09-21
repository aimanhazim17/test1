# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, get_data_from_ceic
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
# Country panel
seriesids_all = pd.read_csv(path_ceic + "ceic_macro_quarterly" + ".csv")
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
    # collapse into quarterly
    df_sub["quarter"] = pd.to_datetime(df_sub["date"]).dt.to_period("q")  # quarterly
    df_sub = df_sub.groupby(["quarter", "country"])[col].mean().reset_index(drop=False)
    df_sub = df_sub[["quarter", "country", col]]
    # merge
    if count_col == 0:
        df = df_sub.copy()
    elif count_col > 0:
        df = df.merge(df_sub, on=["quarter", "country"], how="outer", validate="one_to_one")
    # next
    count_col += 1
df = df.reset_index(drop=True)
# Brent
df_brent = get_data_from_ceic(
    series_ids=[42651501],
    start_date=t_start,
    historical_extension=True
)
df_brent = df_brent[["date", "value"]]
df_brent = df_brent.rename(columns={"date": "date", "value": "brent"})
df_brent["quarter"] = pd.to_datetime(df_brent["date"]).dt.to_period("q")  # quarterly
df_brent = df_brent.groupby("quarter")["brent"].mean().reset_index(drop=False)
# Merge
df = df.merge(df_brent, on="quarter", how="left", validate="many_to_one")
# save interim copy
df["quarter"] = df["quarter"].astype("str")
df.to_parquet(path_data + "data_macro_quarterly_raw" + ".parquet")

# %%
# II --- Wrangle
# Read downloaded data
df = pd.read_parquet(path_data + "data_macro_quarterly_raw" + ".parquet")
# Set groupby cols
cols_groups = ["country", "quarter"]
# Sort
df = df.sort_values(by=cols_groups, ascending=[True, True])
# Reset indices
df = df.reset_index(drop=True)
# Save processed output
df.to_parquet(path_data + "data_macro_quarterly" + ".parquet")

# %%
# X --- Notify
telsendmsg(conf=tel_config, msg="global-plucking --- compile_data_macro_quarterly: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%