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
import analysis_plucking_ugap_quarterly
import descriptive_plucking_ugap_quarterly_viz
# %%
import analysis_macrodynamics_ugap
import analysis_macrodynamics_ugap_ae
import analysis_macrodynamics_ugap_eme

# %%
# X --- Notify
telsendmsg(conf=tel_config, msg="global-plucking --- main_ugap_quarterly: COMPLETED")

# End
print("\n----- Ran in " + "{:.0f}".format(time.time() - time_start) + " seconds -----")

# %%
