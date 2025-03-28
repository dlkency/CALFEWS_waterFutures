import numpy as np
import pandas as pd
import h5py
from calfews_src.util import MGHMM_generate_trace_flexible_start,  regularize_covariance 
import sys
from datetime import datetime

start_time = datetime.now()

### results folder and number of MC samples & DU samples from command line
MGHMMdir = 'calfews_src/data/MGHMM_synthetic/start_year/'
filestart = 'DailyQ_s'
annualfilestart = 'AnnualQ_s'
fileend = '.csv'
numMC = 100

udict = {'dry_state_mean_multiplier': 1, 'wet_state_mean_multiplier': 1,\
         'covariance_matrix_dry_multiplier': 1, 'covariance_matrix_wet_multiplier': 1, \
         'transition_drydry_addition': 0, 'transition_wetwet_addition': 0}

nYears = 4
all_states = []
closest_yr = []

for start_year in range(1996, 2025):

  for mc in range(numMC):
    udict['synth_gen_seed'] = mc
    annual, df, binary_states, closest_year  = MGHMM_generate_trace_flexible_start(nYears, udict, start_year = start_year, drop_date=False)
    df.to_csv(f"{MGHMMdir}{filestart}_{start_year}_{mc}{fileend}", index=False)
    np.savetxt(f"{MGHMMdir}{annualfilestart}_{start_year}_{mc}{fileend}", annual, delimiter=',')
      
    all_states.append({
              'MC_Sample': mc,
              'Year1_State': binary_states[0],
              'Year2_State': binary_states[1] if nYears >=2 else np.nan,
              'Year3_State': binary_states[2] if nYears >=3 else np.nan,
              'Year4_State': binary_states[3] if nYears >=4 else np.nan
              # Add more years if nYears >3
          })
    closest_yr.append({
              'MC_Sample': mc,
              'Year1_State': closest_year[0],
              'Year2_State': closest_year[1] if nYears >=2 else np.nan,
              'Year3_State': closest_year[2] if nYears >=3 else np.nan,
              'Year4_State': closest_year[3] if nYears >=4 else np.nan
              # Add more years if nYears >3
          })
    
  states_df = pd.DataFrame(all_states)
  states_filename = f"{MGHMMdir}All_States_summary_{start_year}.csv"
  states_df.to_csv(states_filename, index=False)
  closest_year_df = pd.DataFrame(closest_yr)
  states_filename = f"{MGHMMdir}closest_year_df_summary_{start_year}.csv"
  closest_year_df.to_csv(states_filename, index=False)