#######################code to generate MGHMM synthetic data with flexible start year########################
#######################leap years are not handled############################################################

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

nYears = 30
all_states = []
closest_yr = []

for start_year in range(1996, 2025):
    for mc in range(numMC):
        udict['synth_gen_seed'] = mc
        annual, df, binary_states, closest_year = MGHMM_generate_trace_flexible_start(
            nYears, udict, start_year=start_year, drop_date=False
        )
        df.to_csv(f"{MGHMMdir}{filestart}_{start_year}_{mc}{fileend}", index=False)
        np.savetxt(f"{MGHMMdir}{annualfilestart}_{start_year}_{mc}{fileend}", annual, delimiter=',')
        
        # Create state dictionary dynamically for the number of years
        state_dict = {'MC_Sample': mc}
        closest_year_dict = {'MC_Sample': mc}
        
        for year in range(nYears):
            state_key = f'Year{year+1}_State'
            closest_key = f'Year{year+1}_Closest'
            if year < len(binary_states):
                state_dict[state_key] = binary_states[year]
                closest_year_dict[closest_key] = closest_year[year]
            else:
                state_dict[state_key] = np.nan
                closest_year_dict[closest_key] = np.nan
        
        all_states.append(state_dict)
        closest_yr.append(closest_year_dict)
    
    states_df = pd.DataFrame(all_states)
    states_filename = f"{MGHMMdir}All_States_summary_{start_year}.csv"
    states_df.to_csv(states_filename, index=False)
    
    closest_year_df = pd.DataFrame(closest_yr)
    closest_year_filename = f"{MGHMMdir}closest_year_df_summary_{start_year}.csv"
    closest_year_df.to_csv(closest_year_filename, index=False)