# %%
from helper import telsendmsg, telsendfiles, telsendimg
import time
import os
from dotenv import load_dotenv

time_start = time.time()

# %%
# 0 --- Main settings
load_dotenv()
tel_config = os.getenv("TEL_CONFIG")

# %%
# I --- Which scripts to run in succession?
# %%
import descriptive_plucking_urate_quarterly_bizcycles
# %%
import analysis_plucking_ugap_quarterly
import descriptive_plucking_ugap_quarterly_viz
# %%
import descriptive_plucking_ugap_quarterly_stylisedstats
# %%
import descriptive_urate_quarterly_stylisedstats
# %%
import analysis_phillipscurve_ugap
import analysis_phillipscurve_ugap_ae
import analysis_phillipscurve_ugap_eme
import analysis_phillipscurve_ugap_cbyc
# %%
import analysis_phillipscurve_urate
import analysis_phillipscurve_urate_ae
import analysis_phillipscurve_urate_eme
import analysis_phillipscurve_urate_cbyc
# %%
import analysis_phillipscurve_urate_smoothed
import analysis_phillipscurve_urate_smoothed_ae
import analysis_phillipscurve_urate_smoothed_eme
import analysis_phillipscurve_urate_smoothed_cbyc
# %%
import analysis_macrodynamics_rgdp_threshold
import analysis_macrodynamics_rgdp_threshold_ae
import analysis_macrodynamics_rgdp_threshold_eme
# %%
import analysis_macrodynamics_urate_threshold
import analysis_macrodynamics_urate_threshold_ae
import analysis_macrodynamics_urate_threshold_eme
# %%
import analysis_macrodynamics_rgdp_fd_threshold
import analysis_macrodynamics_rgdp_fd_threshold_ae
import analysis_macrodynamics_rgdp_fd_threshold_eme
# %%
import analysis_macrodynamics_urate_fd_threshold
import analysis_macrodynamics_urate_fd_threshold_ae
import analysis_macrodynamics_urate_fd_threshold_eme
# %%
import analysis_macrodynamics_ugap
import analysis_macrodynamics_ugap_channels
import analysis_macrodynamics_ugap_ae
import analysis_macrodynamics_ugap_channels_ae
import analysis_macrodynamics_ugap_eme
import analysis_macrodynamics_ugap_channels_eme
# %%
import analysis_macrodynamics_urate
import analysis_macrodynamics_urate_ae
import analysis_macrodynamics_urate_eme
# %%
import analysis_macrodynamics_rgdp
import analysis_macrodynamics_rgdp_ae
import analysis_macrodynamics_rgdp_eme


# %%
# X --- Notify
telsendmsg(conf=tel_config, msg="global-plucking --- main: COMPLETED")

# %%
# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
