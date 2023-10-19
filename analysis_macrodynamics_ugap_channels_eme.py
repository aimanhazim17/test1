# %%
import pandas as pd
from datetime import date, timedelta
import re
from helper import telsendmsg, telsendimg, telsendfiles, pil_img2pdf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# %%
# II --- Pre-analysis wrangling
# Trim countries
list_countries_keep = [
    # "australia",
    "malaysia",
    # "singapore",
    "thailand",
    "indonesia",  # no urate data
    # "philippines",  # no urate data
    # "united_states",  # problems with BER
    # "united_kingdom",
    # "germany",
    # "france",
    # "italy",
    # "japan",
    # "south_korea",
    # "taiwan",  # not covered country
    # "hong_kong_sar_china_",
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
# III --- Set up
# Setup
endog_base = ["expcpi", "privdebt", "urate_gap_ratio", "corecpi", "stir", "reer"]
colours_all_endog = [
    "red",
    "darkgreen",
    "black",
    "darkblue",
    "darkred",
    "cadetblue",
    "grey",
]  # +1 for "own"
exog_base = ["brent", "gepu", "maxgepu"]
# Parameters for charts
fig_max_cols = len(endog_base)
fig_max_rows = len(endog_base)
# Generate list of list of endog and exog variables to rotate between
nested_list_endog = []
nested_list_exog = []
for col_endog_loc in range(len(endog_base)):
    # endogs
    sublist_endog = endog_base[:col_endog_loc] + endog_base[col_endog_loc + 1 :]
    nested_list_endog.append(sublist_endog)  # generate list of list
    # exogs
    sublist_exog = exog_base.copy()
    sublist_exog = sublist_exog + [endog_base[col_endog_loc]]
    nested_list_exog.append(sublist_exog)  # generate list of list

# %%
# IV --- Estimate LPX and generate IRF decomp frame
count_irf = 0
for list_endog, list_exog in tqdm(zip(nested_list_endog, nested_list_exog)):
    # estimate LPX model
    irf = lp.PanelLPX(
        data=df,
        Y=list_endog,
        X=list_exog,
        response=list_endog,
        horizon=16,
        lags=4,
        varcov="robust",
        ci_width=0.95,
    )
    # remove CIs from IRFs
    for col in ["UB", "LB"]:
        del irf[col]
    # rename irf
    irf = irf.rename(columns={"Mean": list_exog[-1]})
    # consolidate
    if count_irf == 0:
        irf_consol = irf.copy()
    elif count_irf > 0:
        irf_consol = irf_consol.merge(
            irf, on=["Shock", "Response", "Horizon"], how="outer"
        )
    # next
    count_irf += 1
# Load reference IRF from another script
irf_total = pd.read_parquet(path_output + "macrodynamics_ugap_lp_irf_eme" + ".parquet")
# Rename reference IRFs
irf_total = irf_total.rename(columns={"Mean": "Total"})
# Remove CIs from reference IRFs
for col in ["UB", "LB"]:
    del irf_total[col]
# Merge with LPX frame
irf_consol = irf_consol.merge(
    irf_total, on=["Shock", "Response", "Horizon"], how="outer"
)
# Compute channel sizes
for col in endog_base:
    irf_consol[col] = irf_consol["Total"] - irf_consol[col]
irf_consol["Own"] = irf_consol["Total"] - irf_consol[endog_base].sum(axis=1)
# Generate output
irf_consol.to_parquet(path_output + "macrodynamics_ugap_lp_irf_channels_eme" + ".parquet")


# %%
# V --- Plot IRFs
# Define channels function
def IRFPlotChannels(
    irf,
    response,
    shock,
    channels,
    channel_colours,
    n_columns,
    n_rows,
    maintitle="Local Projections Model: Propagation Channels",
    show_fig=False,
    save_pic=False,
    out_path="",
    out_name="",
):
    if (len(response) * len(shock)) > (n_columns * n_rows):
        raise NotImplementedError(
            "Number of subplots (n_columns * n_rows) is smaller than number of IRFs to be plotted (n)"
        )
    # Set number of rows and columns
    n_col = n_columns
    n_row = n_rows
    # Generate titles first
    list_titles = []
    for r in response:
        for s in shock:
            subtitle = [s + " -> " + r]
            list_titles = list_titles + subtitle
    # Main plot settings
    fig = make_subplots(rows=n_row, cols=n_col, subplot_titles=list_titles)
    # Subplot loops
    count_col = 1
    count_row = 1
    legend_count = 0
    for r in response:
        for s in shock:
            d = irf.loc[(irf["Response"] == r) & (irf["Shock"] == s)]
            d["Zero"] = 0  # horizontal line
            # Set legend
            if legend_count == 0:
                showlegend_bool = True
            elif legend_count > 0:
                showlegend_bool = False
            # Zero
            fig.add_trace(
                go.Scatter(
                    x=d["Horizon"],
                    y=d["Zero"],
                    mode="lines",
                    line=dict(color="grey", width=1, dash="solid"),
                    showlegend=showlegend_bool,
                ),
                row=count_row,
                col=count_col,
            )
            # Total
            fig.add_trace(
                go.Scatter(
                    x=d["Horizon"],
                    y=d["Total"],
                    mode="lines",
                    line=dict(color="black", width=3, dash="solid"),
                    showlegend=showlegend_bool,
                ),
                row=count_row,
                col=count_col,
            )
            # Add channels
            for c, c_colour in zip(channels, channel_colours):
                fig.add_trace(
                    go.Bar(
                        x=d["Horizon"],
                        y=d[c],
                        name=c,
                        marker=dict(color=c_colour),
                        showlegend=showlegend_bool,
                    ),
                    row=count_row,
                    col=count_col,
                )
            count_col += 1  # move to next
            if count_col <= n_col:
                pass
            elif count_col > n_col:
                count_col = 1
                count_row += 1
            # No further legends
            legend_count += 1
    fig.update_annotations(font_size=11)
    fig.update_layout(
        title=maintitle,
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        barmode="relative",
        font=dict(color="black", size=11),
    )
    if show_fig == True:
        fig.show()
    if save_pic == True:
        fig.write_image(out_path + out_name + ".png", height=1080, width=1920)
        fig.write_html(out_path + out_name + ".html")
    return fig


# Load interim dataframe (so we can block run this script)
irf_consol = pd.read_parquet(
    path_output + "macrodynamics_ugap_lp_irf_channels_eme" + ".parquet"
)

# All IRFs and channels
fig_irf = IRFPlotChannels(
    irf=irf_consol,
    response=endog_base,
    shock=endog_base,
    channels=endog_base + ["Own"],
    channel_colours=colours_all_endog,
    n_columns=fig_max_cols,
    n_rows=fig_max_rows,
    maintitle="IRFs" + " (EMEs)",
    show_fig=False,
    save_pic=False,
)
file_name = path_output + "macrodynamics_ugap_lp_irf_channels_eme"
fig_irf.write_image(file_name + ".png", height=1080, width=1920)
telsendimg(conf=tel_config, path=file_name + ".png", cap=file_name)

# Shock by shock
pic_names = []
for shock in tqdm(endog_base):
    pic_name = path_output + "macrodynamics_ugap_lp_irf_channels_eme" + "_" + shock
    pic_names = pic_names + [pic_name]
    fig_irf = IRFPlotChannels(
        irf=irf_consol,
        response=endog_base,
        shock=[shock],
        channels=endog_base + ["Own"],
        channel_colours=colours_all_endog,
        n_columns=-1 * (-1 * len(endog_base) // 3),
        n_rows=len(endog_base) // 2,
        maintitle="Responses from " + shock + " Shock" + " (EMEs)",
        show_fig=False,
        save_pic=False,
    )
    fig_irf.write_image(
        pic_name + ".png",
        height=1080,
        width=1920,
    )
file_name = path_output + "macrodynamics_ugap_lp_irf_channels_eme"
pil_img2pdf(
    list_images=pic_names,
    extension="png",
    pdf_name=file_name,
)
telsendfiles(
    conf=tel_config,
    path=file_name + ".pdf",
    cap=file_name,
)

# %%
# X --- Notify
telsendmsg(
    conf=tel_config,
    msg="global-plucking --- analysis_macrodynamics_ugap_channels_eme: COMPLETED",
)

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
