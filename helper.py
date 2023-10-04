import pandas as pd
import telegram_send
from linearmodels import PanelOLS, RandomEffects
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.ar_model import ar_select_order
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import date
from PIL import Image
from ceic_api_client.pyceic import Ceic
import re
import os
import requests
import json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
plt.switch_backend("agg")


# --- Notifications


def telsendimg(conf="", path="", cap=""):
    with open(path, "rb") as f:
        telegram_send.send(conf=conf, images=[f], captions=[cap])


def telsendfiles(conf="", path="", cap=""):
    with open(path, "rb") as f:
        telegram_send.send(conf=conf, files=[f], captions=[cap])


def telsendmsg(conf="", msg=""):
    telegram_send.send(conf=conf, messages=[msg])


# --- Data
def get_data_from_api_ceic(
    series_ids: list[float],
    series_names: list[str],
    start_date: date,
    historical_extension: bool = False,
) -> pd.DataFrame:
    """
    Get CEIC data.
    Receive a list of series IDs (e.g., [408955597] for CPI inflation YoY) from CEIC
    and output a pandas data frame (for single entity time series).

    :series_ids `list[float]`: a list of CEIC Series IDs\n
    :start_date `date`: a date() object of the start date e.g. date(1991, 1, 1)\n
    "continuous `Optional[boolean]`: When set to true, series will include extended historical timepoints\n
    :return `pd.DataFrame`: A DataFrame instance of the data
    """
    df = pd.DataFrame()
    series_list = ",".join(map(str, series_ids))

    if historical_extension == False:
        PATH = f"https://api.ceicdata.com/v2//series/{series_list}/data?format=json&start_date={start_date}"
        response = requests.get(f"{PATH}&token={os.getenv('CEIC_API_KEY')}")
        content = json.loads(response.text)["data"]
    else:
        content = []
        for series in series_ids:
            PATH = f"https://api.ceicdata.com/v2//series/{series}/data?format=json&start_date={start_date}"
            response = requests.get(
                f"{PATH}&with_historical_extension=True&token={os.getenv('CEIC_API_KEY')}"
            )
            content = content + json.loads(response.text)["data"]
    for i, j in zip(
        range(len(series_ids)), series_names
    ):  # series names not in API json
        data = pd.DataFrame(content[i]["timePoints"])[["date", "value"]]
        # name = content[i]["layout"][0]["table"]["name"]
        name = "_".join(j.split(": ")[1:])  # name --> j
        data["name"] = re.sub("[^A-Za-z0-9]+", "_", name).lower()
        country = content[i]["layout"][0]["topic"]["name"]  # section --> topic
        data["country"] = re.sub("[^A-Za-z0-9]+", "_", country).lower()
        df = pd.concat([df, data])

    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    return df


def get_data_from_ceic(
    series_ids: list[float], start_date: date, historical_extension: bool = False
) -> pd.DataFrame:
    """
    Get CEIC data.
    Receive a list of series IDs (e.g., [408955597] for CPI inflation YoY) from CEIC
    and output a pandas data frame (for single entity time series).

    :series_ids `list[float]`: a list of CEIC Series IDs\n
    :start_date `date`: a date() object of the start date e.g. date(1991, 1, 1)\n
    :return `pd.DataFrame`: A DataFrame instance of the data
    """
    Ceic.login(username=os.getenv("CEIC_USERNAME"), password=os.getenv("CEIC_PASSWORD"))

    df = pd.DataFrame()
    content = []
    if not historical_extension:
        content = Ceic.series(series_id=series_ids, start_date=start_date).data
    else:
        for series in tqdm(series_ids):
            try:
                content += Ceic.series(
                    series_id=series,
                    start_date=start_date,
                    with_historical_extension=True,
                ).data
            except:
                # revert to without historical extension if fails
                content += Ceic.series(
                    series_id=series,
                    start_date=start_date,
                    with_historical_extension=False,
                ).data
    for i in range(len(series_ids)):  # for i in range(len(content))
        data = pd.DataFrame(
            [(tp._date, tp.value) for tp in content[i].time_points],
            columns=["date", "value"],
        )
        data["name"] = re.sub("[^A-Za-z0-9]+", "_", content[i].metadata.name).lower()
        data["country"] = re.sub(
            "[^A-Za-z0-9]+", "_", content[i].metadata.country.name
        ).lower()
        df = pd.concat([df, data])
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    return df


# --- ITS


def estimate_its_arx(
    data: pd.DataFrame,
    col_y: str,
    shock_timing: str,
    alpha_choice: str,
    z_score_for_ci: float,
    maxlag_choice: int,
    col_x=None,
):
    # prelims
    df = data.copy()

    # split into pre, post, and full data
    df_pre = df[df.index < shock_timing].copy()
    df_post = df[df.index >= shock_timing].copy()

    # exogenous variables
    if col_x is None:
        exog_pre = None
        exog_pre_values = None

        # exog_post = None
        exog_post_values = None
    elif col_x is not None:
        exog_pre = df_pre[col_x].copy()
        exog_pre_values = df_pre[col_x].values

        # exog_post = df_post[col_x].copy()
        exog_post_values = df_post[col_x].values

    # find order
    ar_order = ar_select_order(
        df_pre[col_y], maxlag=maxlag_choice, trend=alpha_choice, ic="hqic"
    ).ar_lags
    if ar_order is None:
        ar_order = 1  # fail-safe, defaults to AR(1)-X if lag selection is indifferent
    elif ar_order is not None:
        ar_order = ar_order[0]

    # estimate model
    est_arx = smt.AutoReg(
        endog=df_pre[col_y],
        exog=exog_pre,
        trend=alpha_choice,  # alpha
        lags=range(1, ar_order + 1),
    )
    res_arx = est_arx.fit(cov_type="HAC", cov_kwds={"maxlags": ar_order})

    # pre-event prediction (pre-event period; with interval)
    pred = res_arx.get_prediction(
        start=0, end=len(df_pre) - 1, dynamic=False, exog=exog_pre_values
    )  # zero-indexed
    pred = pd.concat([pred.predicted_mean, pred.se_mean], axis=1)
    pred = pred.reset_index().rename(
        columns={"index": "month", "predicted_mean": "ptpred_" + col_y}
    )
    pred = pred.set_index("month")
    pred["lbpred_" + col_y] = pred["ptpred_" + col_y] - z_score_for_ci * pred["mean_se"]
    pred["ubpred_" + col_y] = pred["ptpred_" + col_y] + z_score_for_ci * pred["mean_se"]
    del pred["mean_se"]

    # post-event forecast (counterfactual)
    fcast = res_arx.get_prediction(
        start=len(df_pre),  # zero-indexed
        end=len(df_pre) + len(df_post) - 1,
        dynamic=True,
        exog_oos=exog_post_values,
    )
    fcast = pd.concat([fcast.predicted_mean, fcast.se_mean], axis=1)
    fcast = fcast.reset_index().rename(
        columns={"index": "month", "predicted_mean": "ptfcast_" + col_y}
    )
    fcast = fcast.set_index("month")
    fcast["lbfcast_" + col_y] = (
        fcast["ptfcast_" + col_y] - z_score_for_ci * fcast["mean_se"]
    )
    fcast["ubfcast_" + col_y] = (
        fcast["ptfcast_" + col_y] + z_score_for_ci * fcast["mean_se"]
    )
    del fcast["mean_se"]

    # merge everything
    df_final = pd.concat([df[col_y], fcast, pred], axis=1)

    # compute ITS impact
    df_its = df_final[df_final.index >= shock_timing].copy()
    df_its["its"] = df_its[col_y] - df_its["ptfcast_" + col_y]
    df_its["its_lb"] = df_its[col_y] - df_its["ubfcast_" + col_y]  # flipped
    df_its["its_ub"] = df_its[col_y] - df_its["lbfcast_" + col_y]  # flipped
    df_its = df_its[["its", "its_lb", "its_ub"]]

    # output
    return df_final, df_its


# --- Linear regressions

# --- TIME SERIES MODELS


def est_varx(
    df: pd.DataFrame,
    cols_endog: list,
    run_varx: bool,
    cols_exog: list,
    choice_ic: str,
    choice_trend: str,
    choice_horizon: int,
    choice_maxlags: int,
):
    # Work on copy
    d = df.copy()

    # Estimate model
    if run_varx:
        mod = smt.VAR(endog=d[cols_endog], exog=d[cols_exog])
    if not run_varx:
        mod = smt.VAR(endog=d[cols_endog])
    res = mod.fit(ic=choice_ic, trend=choice_trend, maxlags=choice_maxlags)
    irf = res.irf(periods=choice_horizon)

    # Output
    return res, irf


# --- CHARTS


def heatmap(
    input: pd.DataFrame,
    mask: bool,
    colourmap: str,
    outputfile: str,
    title: str,
    lb: float,
    ub: float,
    format: str,
    show_annot: bool,
    y_fontsize: float,
    x_fontsize: float,
    title_fontsize: float,
):
    fig = plt.figure()
    sns.heatmap(
        input,
        mask=mask,
        annot=show_annot,
        cmap=colourmap,
        center=0,
        annot_kws={"size": 9},  # 9, 12, 16, 20, 24, 28
        vmin=lb,
        vmax=ub,
        xticklabels=True,
        yticklabels=True,
        fmt=format,
    )
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=x_fontsize)
    plt.yticks(fontsize=y_fontsize)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close()
    return fig


def heatmap_layered(
    actual_input: pd.DataFrame,
    disp_input: pd.DataFrame,
    mask: bool,
    colourmap: str,
    outputfile: str,
    title: str,
    lb: float,
    ub: float,
    format: str,
):
    fig = plt.figure()
    sns.heatmap(
        actual_input,
        mask=mask,
        annot=disp_input,
        cmap=colourmap,
        center=0,
        annot_kws={"size": 12},  # 9, 12, 20, 28
        vmin=lb,
        vmax=ub,
        xticklabels=True,
        yticklabels=True,
        fmt=format,
    )
    plt.title(title, fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    fig.savefig(outputfile)
    plt.close()
    return fig


def pil_img2pdf(list_images: list, extension: str, pdf_name: str):
    seq = list_images.copy()  # deep copy
    list_img = []
    file_pdf = pdf_name + ".pdf"
    run = 0
    for i in seq:
        img = Image.open(i + "." + extension)
        img = img.convert("RGB")  # PIL cannot save RGBA files as pdf
        if run == 0:
            first_img = img.copy()
        elif run > 0:
            list_img = list_img + [img]
        run += 1
    first_img.save(
        file_pdf, "PDF", resolution=100.0, save_all=True, append_images=list_img
    )


def boxplot(
    data: pd.DataFrame,
    y_cols: list,
    x_col: str,
    trace_names: list,
    colours: list,
    main_title: str,
):
    # prelims
    d = data.copy()
    # generate figure
    fig = go.Figure()
    # add box plots one by one
    max_candidates = []
    min_candidates = []
    for y, trace_name, colour in zip(y_cols, trace_names, colours):
        fig.add_trace(
            go.Box(
                y=d[y],
                x=d[x_col],
                name=trace_name,
                marker=dict(opacity=0, color=colour),
                boxpoints="outliers",
            )
        )
        max_candidates = max_candidates + [d[y].quantile(q=0.99)]
        min_candidates = min_candidates + [d[y].min()]
    fig.update_yaxes(
        range=[min(min_candidates), max(max_candidates)],
        showgrid=True,
        gridwidth=1,
        gridcolor="grey",
    )
    # layouts
    fig.update_layout(
        title=main_title, plotbg_color="white", boxmode="group", height=768, width=1366
    )
    fig.update_xaxes(categoryorder="category ascending")
    # output
    return fig


def boxplot_time(
    data: pd.DataFrame,
    y_col: str,
    x_col: str,
    t_col: str,
    colours: list,
    main_title: str,
):
    # prelims
    d = data.copy()
    list_t = list(d[t_col].unique())
    list_t.sort()
    # generate figure
    fig = go.Figure()
    # add box plots one by one
    max_candidates = []
    min_candidates = []
    for t, colour in zip(list_t, colours):
        fig.add_trace(
            go.Box(
                y=d.loc[d[t_col] == t, y_col],
                x=d.loc[d[t_col] == t, x_col],
                name=str(t),
                marker=dict(opacity=0, color=colour),
                boxpoints="outliers",
            )
        )
        max_candidates = max_candidates + [d.loc[d[t_col] == t, y_col].quantile(q=0.99)]
        min_candidates = min_candidates + [d.loc[d[t_col] == t, y_col].min()]
    fig.update_yaxes(
        range=[min(min_candidates), max(max_candidates)],
        showgrid=True,
        gridwidth=1,
        gridcolor="grey",
    )
    # layouts
    fig.update_layout(
        title=main_title,
        plot_bgcolor="white",
        boxmode="group",
        font=dict(color="black", size=12),
        height=768,
        width=1366,
    )
    fig.update_xaxes(categoryorder="category ascending")
    # output
    return fig


def barchart(
    data: pd.DataFrame, y_col: str, x_col: str, main_title: str, decimal_points: int
):
    # generate figure
    fig = go.Figure()
    # add bar chart
    fig.add_trace(
        go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker=dict(color="lightblue"),
            text=data[y_col].round(decimal_points).astype("str"),
            textposition="outside",
        )
    )
    # layouts
    fig.update_layout(
        title=main_title,
        plot_bgcolor="white",
        font=dict(color="black", size=16),
        height=768,
        width=1366,
    )
    fig.update_traces(textfont_size=22)
    # output
    return fig


def wide_grouped_barchart(
    data: pd.DataFrame,
    y_cols: list,
    group_col: str,
    main_title: str,
    decimal_points: int,
    group_colours: list,
    custom_ymin=None,
    custom_ymax=None,
):
    # generate figure
    fig = go.Figure()
    # add bar chart
    for group, colour in zip(list(data[group_col].unique()), group_colours):
        x_vals = list(data[y_cols].columns)
        y_vals = list(data.loc[data[group_col] == group, y_cols].values[0])
        y_str = [str(round(i, decimal_points)) for i in y_vals]
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=str(group),
                marker=dict(color=colour),
                text=y_str,
                textposition="outside",
            )
        )
    # layouts
    fig.update_layout(
        title=main_title,
        plot_bgcolor="white",
        font=dict(color="black", size=16),
        height=768,
        width=1366,
    )
    fig.update_traces(textfont_size=22)
    if (custom_ymax is not None) & (custom_ymin is not None):
        fig.update_yaxes(range=[custom_ymin, custom_ymax])
    # output
    return fig


def manual_irf_subplots(
    data,
    endog,
    shock_col,
    response_col,
    irf_col,
    horizon_col,
    main_title,
    maxrows,
    maxcols,
    line_colour,
    annot_size,
    font_size,
):
    # Create titles first
    titles = []
    for response in endog:
        for shock in endog:
            titles = titles + [shock + " -> " + response]
    maxr = maxrows
    maxc = maxcols
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=titles)
    nr = 1
    nc = 1
    # columns: shocks, rows: responses; move columns, then rows
    for response in endog:
        for shock in endog:
            # Data copy
            d = data[
                (data[shock_col] == shock) & (data[response_col] == response)
            ].copy()
            # Add selected series
            fig.add_trace(
                go.Scatter(
                    x=d[horizon_col].astype("str"),
                    y=d[irf_col],
                    mode="lines",
                    line=dict(width=3, color=line_colour),
                ),
                row=nr,
                col=nc,
            )
            # Add zero line
            fig.add_hline(
                y=0, line_width=1, line_dash="solid", line_color="grey", row=nr, col=nc
            )
            # Move to next subplot
            nc += 1
            if nr > maxr:
                raise NotImplementedError(
                    "More subplots than allowed by dimension of main plot!"
                )
            if nc > maxc:
                nr += 1  # next row
                nc = 1  # reset column
    for annot in fig["layout"]["annotations"]:
        annot["font"] = dict(size=annot_size, color="black")  # subplot title font size
    fig.update_layout(
        title=main_title,
        # yaxis_title=y_title,
        plot_bgcolor="white",
        hovermode="x",
        font=dict(color="black", size=font_size),
        showlegend=False,
        height=768,
        width=1366,
    )
    # output
    return fig


def manual_irf_subplots_channels(
    data,
    shocks,
    responses,
    shock_col,
    response_col,
    main_irf_col,
    channels_cols,
    horizon_col,
    main_title,
    maxrows,
    maxcols,
    line_colour,
    bar_colours,
    annot_size,
    font_size,
):
    # Create titles first
    titles = []
    for response in responses:
        for shock in shocks:
            titles = titles + [shock + " -> " + response]
    maxr = maxrows
    maxc = maxcols
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=titles)
    nr = 1
    nc = 1
    # columns: shocks, rows: responses; move columns, then rows
    legend_count = 0
    for response in responses:
        for shock in shocks:
            # Data copy
            d = data[
                (data[shock_col] == shock) & (data[response_col] == response)
            ].copy()
            # Set legend
            if legend_count == 0:
                showlegend_bool = True
            elif legend_count > 0:
                showlegend_bool = False
            # Add total impact (main irf)
            fig.add_trace(
                go.Scatter(
                    x=d[horizon_col].astype("str"),
                    y=d[main_irf_col],
                    name="total",
                    mode="lines",
                    line=dict(width=3, color=line_colour),
                    showlegend=showlegend_bool,
                ),
                row=nr,
                col=nc,
            )
            # Add channels
            for col, colour in zip(channels_cols, bar_colours):
                fig.add_trace(
                    go.Bar(
                        x=d[horizon_col].astype("str"),
                        y=d[col],
                        name=col,
                        marker=dict(color=colour),
                        showlegend=showlegend_bool,
                    ),
                    row=nr,
                    col=nc,
                )
            # Add zero line
            fig.add_hline(
                y=0, line_width=1, line_dash="solid", line_color="grey", row=nr, col=nc
            )
            # Move to next subplot
            nc += 1
            if nr > maxr:
                raise NotImplementedError(
                    "More subplots than allowed by dimension of main plot!"
                )
            if nc > maxc:
                nr += 1  # next row
                nc = 1  # reset column
            # No further legends
            legend_count += 1
    for annot in fig["layout"]["annotations"]:
        annot["font"] = dict(size=annot_size, color="black")  # subplot title font size
    fig.update_layout(
        title=main_title,
        # yaxis_title=y_title,
        plot_bgcolor="white",
        hovermode="x",
        font=dict(color="black", size=font_size),
        showlegend=True,
        barmode="relative",
        height=768,
        width=1366,
    )
    # output
    return fig


def subplots_linecharts(
        data: pd.DataFrame,
        col_group: str,
        cols_values: list[str],
        cols_values_nice: list[str],
        col_time: str,
        annot_size: list[int],
        font_size: list[int],
        line_colours: list[str], 
        line_dashes: list[str],
        main_title: str,
        maxrows: int,
        maxcols: int,
):
    # Create titles first
    titles = []
    for group in list(data[col_group].unique()):
        titles = titles + [group]
    maxr = maxrows
    maxc = maxcols
    fig = make_subplots(rows=maxr, cols=maxc, subplot_titles=titles)
    nr = 1
    nc = 1
    # columns: shocks, rows: responses; move columns, then rows
    legend_count = 0
    for group in list(data[col_group].unique()):
        # Data copy
        d = data[(data[col_group] == group)].copy()
        # Set legend
        if legend_count == 0:
            showlegend_bool = True
        elif legend_count > 0:
            showlegend_bool = False
        # Add line plots 
        for col, col_nice, line_colour, line_dash in zip(cols_values, cols_values_nice, line_colours, line_dashes):
            fig.add_trace(
                go.Scatter(
                    x=d[col_time].astype("str"),
                    y=d[col],
                    name=col_nice,
                    mode="lines",
                    line=dict(width=1.5, color=line_colour, dash=line_dash),
                    showlegend=showlegend_bool,
                ),
                row=nr,
                col=nc,
            )
        # Move to next subplot
        nc += 1
        if nr > maxr:
            raise NotImplementedError(
                "More subplots than allowed by dimension of main plot!"
            )
        if nc > maxc:
            nr += 1  # next row
            nc = 1  # reset column
        # No further legends
        legend_count += 1
    for annot in fig["layout"]["annotations"]:
        annot["font"] = dict(size=annot_size, color="black")  # subplot title font size
    fig.update_layout(
        title=main_title,
        # yaxis_title=y_title,
        plot_bgcolor="white",
        hovermode="x",
        font=dict(color="black", size=font_size),
        showlegend=True,
        barmode="relative",
        height=768,
        width=1366,
    )
    # output
    return fig