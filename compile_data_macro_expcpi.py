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
# I --- Load data
df = pd.read_excel(path_data + "Consensus_CPI_forecast.xlsx", sheet_name="12m_ahead")

# %%
# II --- Wrangle into monthly frame
# Column labels in wide format
dict_rename_wide = {
    "Unnamed: 0": "month",
    'AU': 'australia', 
    'MY': 'malaysia', 
    'SG': 'singapore', 
    'TH': 'thailand', 
    'ID': 'indonesia', 
    'PH': 'philippines', 
    'US': 'united_states', 
    'GB': 'united_kingdom', 
    'DE': 'germany',
    'FR': 'france', 
    'IT': 'italy', 
    'JP': 'japan', 
    'KR': 'south_korea', 
    'HK': 'hong_kong_sar_china_', 
    'IN': 'india', 
    'CH': 'china', 
    'CL': 'chile', 
    'MX': 'mexico', 
    'BR': 'brazil', 
    'XM': 'eurozone'
}
df = df.rename(columns=dict_rename_wide)
# Date format
df["month"] = pd.to_datetime(df["month"]).dt.to_period("m")
df["month"] = df["month"].astype("str")
# Delete eurozone
del df["eurozone"]
# Convert into long form
list_countries = [i for i in df.columns if "month" not in i]
df = pd.melt(
    df,
    id_vars=["month"],
    var_name="country",
    value_vars=list_countries,
    value_name="expcpi"
)
# Sort and reset
df = df.sort_values(by=["country", "month"])
df = df.reset_index(drop=True)
# Output monthly frame
df_m = df.copy()
df_m.to_parquet(path_data + "data_macro_monthly_expcpi.parquet")

# %%
# III --- Wrangle into quarterly frame
# Generate copy
df_q = df_m.copy()
# Generate quarter identifiers
df_q["quarter"] = pd.to_datetime(df_q["month"]).dt.to_period("q")
df_q["quarter"] = df_q["quarter"].astype("str")
# Collapse
df_q = df_q.groupby(["country", "quarter"])["expcpi"].mean().reset_index(drop=False)
# Output quarterly frame
df_q.to_parquet(path_data + "data_macro_quarterly_expcpi.parquet")

# %%
# X --- Notify
telsendmsg(conf=tel_config, msg="global-plucking --- compile_data_macro_expcpi: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%