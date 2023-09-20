import pandas as pd
import numpy as np
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


def compute_urate_floor(
    data,
    levels_labels,
    ref_level_label,
    downturn_threshold,
    bounds_timing_shift,
    hard_bound,
):
    # Deep copy
    df = data.copy()

    # Current column labels
    cols_levels = levels_labels.copy()
    cols_diff = [i + "_diff" for i in cols_levels]
    cols_trough = [i + "_trough" for i in cols_levels]
    cols_peak = [i + "_peak" for i in cols_levels]
    cols_epi = [i + "_epi" for i in cols_levels]
    cols_cepi = [i + "_cepi" for i in cols_levels]  # ceiling episodes
    cols_pace = [i + "_pace" for i in cols_levels]
    cols_ceiling = [i + "_ceiling" for i in cols_levels]

    # Reference column labels
    ref_levels = ref_level_label
    ref_diff = ref_levels + "_diff"
    ref_trough = ref_levels + "_trough"
    ref_peak = ref_levels + "_peak"
    ref_epi = ref_levels + "_epi"
    ref_cepi = ref_levels + "_cepi"
    ref_pace = ref_levels + "_pace"
    ref_ceiling = ref_levels + "_ceiling"

    # Base list of peaks and troughs
    list_peaks = []
    list_troughs = []

    # Store list of months
    list_time = [str(i) for i in list(df["month"])]
    list_indices = [i for i in list(df.index)]

    # Counter for if lower bound or point estimate is being calculated
    count_xbound_now = 0

    # Loop through list of timestamps
    while count_xbound_now <= 1:
        for (
            col_level,
            col_peak,
            col_trough,
            col_epi,
            col_cepi,
            col_pace,
            col_diff,
            col_ceiling,
        ) in zip(
            cols_levels,
            cols_peak,
            cols_trough,
            cols_epi,
            cols_cepi,
            cols_pace,
            cols_diff,
            cols_ceiling,
        ):
            # Add lb as suffix if estimating xbound
            if count_xbound_now == 1:
                col_trough = col_trough + "_lb"
                col_peak = col_peak + "_lb"
                col_epi = col_epi + "_lb"
                col_cepi = col_cepi + "_lb"
                col_pace = col_pace + "_lb"
                col_ceiling = col_ceiling + "_lb"

            # Compute downturn threshold using standard deviation of logdiff
            # threshold_for_this_col = downturn_threshold
            threshold_for_this_col = df[col_level].std() * downturn_threshold
            # threshold_for_this_col = df[col_diff].std() * downturn_threshold

            # Initialise parameters
            t_cp = 0  # peak candidate
            t_ct = 0  # trough candidate
            t_next = t_cp + 1  # initial time stamp
            just_found_peak = False
            just_found_trough = False
            stuck_in_step_one = True
            stuck_in_step_two = True
            stuck_in_step_five = True
            stuck_in_step_six = True

            # Define all requisite steps as functions

            def step_one(just_found_trough: bool, t_cp, t_ct, t_next):
                if just_found_trough:
                    t_cp = t_ct + 1
                    t_next = t_cp + 1
                elif not just_found_trough:
                    t_cp = t_cp + 1
                    t_next = t_next + 1
                return t_cp, t_next

            def step_two(df: pd.DataFrame, t_cp, t_next):
                go_back = (
                    df.loc[df.index == t_cp, col_level].values[0]
                    > df.loc[df.index == t_next, col_level].values[0]  # opposite
                )
                return go_back

            def step_three(df: pd.DataFrame, t_next):
                go_back = (
                    df.loc[df.index == t_cp, col_level].values[0]
                    + threshold_for_this_col
                    > df.loc[df.index == t_next, col_level].values[0]  # opposite
                )
                t_next = t_next + 1  # without changing t_cp
                return go_back, t_next

            def step_four(t_cp, list_peaks):
                list_peaks = list_peaks + [list_time[t_cp]]
                just_found_peak = True
                return list_peaks, just_found_peak

            def step_five(just_found_peak: bool, t_cp, t_ct, t_next):
                if just_found_peak:
                    t_ct = t_cp + 1
                    t_next = t_ct + 1
                elif not just_found_peak:
                    t_ct = t_ct + 1
                    t_next = t_next + 1
                return t_ct, t_next

            def step_six(df: pd.DataFrame, t_ct, t_next):
                go_back = (
                    df.loc[df.index == t_ct, col_level].values[0]
                    < df.loc[df.index == t_next, col_level].values[0]  # opposite
                )
                return go_back

            def step_seven(df: pd.DataFrame, t_next):
                go_back = (
                    df.loc[df.index == t_ct, col_level].values[0]
                    - threshold_for_this_col
                    < df.loc[df.index == t_next, col_level].values[0]  # opposite
                )
                t_next = t_next + 1  # without changing t_cp
                return go_back, t_next

            def step_eight(t_ct, list_troughs):
                list_troughs = list_troughs + [list_time[t_ct]]
                just_found_trough = True
                return list_troughs, just_found_trough

            while t_next <= list_indices[-1]:
                try:
                    # FIND PEAK
                    # step one and two
                    while stuck_in_step_one:
                        # print('Step 1: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_cp, t_next = step_one(
                            just_found_trough=just_found_trough,
                            t_cp=t_cp,
                            t_ct=t_ct,
                            t_next=t_next,
                        )
                        just_found_trough = False  # only allow just_found_trough to be true once per loop
                        stuck_in_step_one = step_two(df=df, t_cp=t_cp, t_next=t_next)
                    stuck_in_step_one = True  # reset so loop will run again
                    # step three
                    while stuck_in_step_two:
                        # print('Step 2-3: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_two, t_next = step_three(
                            df=df, t_next=t_next
                        )  # if true, skips next line
                        restuck_in_step_one = step_two(df=df, t_cp=t_cp, t_next=t_next)
                        while (
                            restuck_in_step_one
                        ):  # if step 3 is executed, but then fails step 2, so back to step 1
                            # print('Back to step 1-2: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_cp, t_next = step_one(
                                just_found_trough=just_found_trough,
                                t_cp=t_cp,
                                t_ct=t_ct,
                                t_next=t_next,
                            )
                            restuck_in_step_one = step_two(
                                df=df, t_cp=t_cp, t_next=t_next
                            )
                    stuck_in_step_two = True  # reset so loop will run again
                    # step four
                    # print('Step 4: t_cp = ' + list_time[t_cp] + ' for ' + col_level)
                    list_peaks, just_found_peak = step_four(
                        t_cp=t_cp, list_peaks=list_peaks
                    )  # we have a peak
                    just_found_peak = True  # voila

                    # FIND TROUGH
                    # step five and six (equivalent to one and two)
                    while stuck_in_step_five:
                        # print('Step 5: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        t_ct, t_next = step_five(
                            just_found_peak=just_found_peak,
                            t_cp=t_cp,
                            t_ct=t_ct,
                            t_next=t_next,
                        )
                        just_found_peak = (
                            False  # only allow just_found_peak to be true once per loop
                        )
                        stuck_in_step_five = step_six(df=df, t_ct=t_ct, t_next=t_next)
                    stuck_in_step_five = True  # reset so loop will run again
                    # step seven (equivalent to three)
                    while stuck_in_step_six:
                        # print('Step 6-7: t_next = ' + list_time[t_next] + ' for ' + col_level)
                        stuck_in_step_six, t_next = step_seven(
                            df=df, t_next=t_next
                        )  # if true, skips next line
                        restuck_in_step_five = step_six(df=df, t_ct=t_ct, t_next=t_next)
                        while restuck_in_step_five:
                            # print('Back to step 5-6: t_next = ' + list_time[t_next] + ' for ' + col_level)
                            t_ct, t_next = step_five(
                                just_found_peak=just_found_peak,
                                t_cp=t_cp,
                                t_ct=t_ct,
                                t_next=t_next,
                            )
                            restuck_in_step_five = step_six(
                                df=df, t_ct=t_ct, t_next=t_next
                            )
                    stuck_in_step_six = True  # reset so loop will run again
                    # step eight (equivalent to four)
                    # print('Step 8: t_ct = ' + list_time[t_ct] + ' for ' + col_level)
                    list_troughs, just_found_trough = step_eight(
                        t_ct=t_ct, list_troughs=list_troughs
                    )  # we have a trough
                except:
                    pass

            # Check
            print("Peaks in " + col_level + ": " + ", ".join(list_peaks))
            print("Troughs in " + col_level + ": " + ", ".join(list_troughs))

            # Add columns indicating peaks and troughs
            df.loc[df["month"].isin(list_peaks), col_peak] = 1
            df[col_peak] = df[col_peak].fillna(0)
            df.loc[df["month"].isin(list_troughs), col_trough] = 1
            df[col_trough] = df[col_trough].fillna(0)

            # For Xbounds
            if count_xbound_now == 1:
                df[col_peak] = df[col_peak].shift(
                    bounds_timing_shift
                )  # move back by h horizons
                df[col_peak] = df[col_peak].fillna(0)

            # Episodes
            df[col_epi] = (df[col_trough] + df[col_peak]).cumsum()
            df.loc[((df[col_trough] == 1) | (df[col_peak] == 1)), col_epi] = (
                df[col_epi] - 1
            )

            # Peak-to-peak episodes
            df[col_cepi] = df[col_peak].cumsum()
            df.loc[df[col_peak] == 1, col_cepi] = (
                df[col_cepi] - 1
            )  # exp start after trough, and end at peaks

            # Calculate average episodic pace
            df[col_pace] = df.groupby(col_epi)[col_diff].transform("mean")
            tab = df.groupby(col_epi)[col_diff].agg("mean").reset_index()
            print(tab)

            # Compute ceiling
            # Check if single expansion
            single_exp = bool(df[col_epi].max() == 0)
            if not single_exp:
                # interpolate
                df.loc[df[col_peak] == 1, col_ceiling] = df[
                    col_level
                ]  # peaks as joints
                try: 
                    df[col_ceiling] = df[col_ceiling].interpolate(
                        method="quadratic"
                    )  # too sparse for cubic
                except:
                    try:
                        df[col_ceiling] = df[col_ceiling].interpolate(
                            method="slinear"
                        )  # too sparse for cubic
                    except:
                        df[col_ceiling] = df[col_ceiling].interpolate(
                            method="linear"
                        )  # too sparse for cubic
                # end-point extrapolation
                cepi_minusone = df[col_cepi].max() - 1
                df["_x"] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_minusone_avgdiff = (
                    df.loc[df[col_cepi] == cepi_minusone, "_x"]
                ).mean()
                del df["_x"]
                nrows_na = len(df.isna())
                for r in range(nrows_na):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = (
                        df[col_ceiling].shift(1) + ceiling_minusone_avgdiff
                    )

                # start-point extrapolation
                df["_x"] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, "_x"]).mean()
                del df["_x"]
                nrows_na = len(df.isna())
                for r in range(nrows_na):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = (
                        df[col_ceiling].shift(-1) - ceiling_one_avgdiff
                    )  # reverse
            if single_exp:
                df[col_peak] = df[ref_peak].copy()
                df[col_cepi] = df[ref_cepi].copy()

                # interpolate
                df.loc[df[col_peak] == 1, col_ceiling] = df[
                    col_level
                ]  # peaks as joints
                try:
                    df[col_ceiling] = df[col_ceiling].interpolate(
                        method="quadratic"
                    )  # too sparse for cubic
                except:
                    try:
                        df[col_ceiling] = df[col_ceiling].interpolate(
                            method="slinear"
                        )  # too sparse for cubic
                    except:
                        df[col_ceiling] = df[col_ceiling].interpolate(
                            method="linear"
                        )  # too sparse for cubic

                # end-point extrapolation
                cepi_minusone = df[col_cepi].max() - 1
                df["_x"] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_minusone_avgdiff = (
                    df.loc[df[col_cepi] == cepi_minusone, "_x"]
                ).mean()
                del df["_x"]
                nrows_na = len(df.isna())
                for r in range(nrows_na):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = (
                        df[col_ceiling].shift(1) + ceiling_minusone_avgdiff
                    )

                # start-point extrapolation
                df["_x"] = df[col_ceiling] - df[col_ceiling].shift(1)
                ceiling_one_avgdiff = (df.loc[df[col_cepi] == 1, "_x"]).mean()
                del df["_x"]
                nrows_na = len(df.isna())
                for r in range(nrows_na):
                    df.loc[df[col_ceiling].isna(), col_ceiling] = (
                        df[col_ceiling].shift(-1) - ceiling_one_avgdiff
                    )  # reverse

            # hard-impose definition of 'ceiling' (floor for urate)
            if hard_bound | (col_level == "ln_lforce") | (col_level == "ln_nks"):
                df.loc[df[col_ceiling] > df[col_level], col_ceiling] = df[
                    col_level
                ]  # replace with levels if first guess is lower; opposite because urate

        # Left-right merge + bounds
        if count_xbound_now == 0:
            df_consol = df.copy()
        elif count_xbound_now == 1:
            df_consol = df_consol.combine_first(df)  # with xbounds

        # Indicator to trigger computation of xbounds
        count_xbound_now += 1

    # Output
    return df
