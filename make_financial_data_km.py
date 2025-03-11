#%%
import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
from itertools import compress
import calfews_src
from calfews_src import *
from calfews_src.visualizer import Visualizer
from calfews_src.util import * 
import seaborn as sns
import os
import time
#%%


sns.set_style(style='whitegrid')

def set_district_keys():
  ##this creates the index to compare PMP district codes with CALFEWS output keys
  district_pmp_keys = {}
  district_pmp_keys['D02'] = 'kerndelta'
  district_pmp_keys['D03'] = 'wheeler'
  district_pmp_keys['D04'] = 'westkern'
  district_pmp_keys['D01'] = 'belridge'
  district_pmp_keys['D05'] = 'berrenda'
  district_pmp_keys['D06'] = 'semitropic'
  district_pmp_keys['D07'] = 'rosedale'
  district_pmp_keys['D08'] = 'buenavista'
  district_pmp_keys['D09'] = 'cawelo'
  district_pmp_keys['D10'] = 'henrymiller'
  district_pmp_keys['D11'] = 'losthills'
  district_pmp_keys['fk13'] = 'tulare'
  district_pmp_keys['fk08'] = 'saucelito'
  district_pmp_keys['fk01'] = 'delano'
  district_pmp_keys['fk06'] = 'lowertule'
  district_pmp_keys['fk03'] = 'kerntulare'
  district_pmp_keys['fk05'] = 'lindsay'
  district_pmp_keys['fk02'] = 'exeter'
  district_pmp_keys['fk04'] = 'lindmore'
  district_pmp_keys['fk07'] = 'porterville'
  district_pmp_keys['fk11'] = 'teapot'
  district_pmp_keys['fk12'] = 'terra'
  district_pmp_keys['fk13'] = 'sosanjoaquin'
  district_pmp_keys['fk09'] = 'shaffer'
  district_pmp_keys['ot2'] = 'northkern'
  district_pmp_keys['ot1'] = 'dudleyridge'
  
  return district_pmp_keys

#%% 
def calculate_district_revenues(df_data, district_display_key, district_pmp_keys, frequencies='AS-OCT'):
    district_prices = pd.read_csv('calfews_src/postprocess/district_water_prices.csv')
    district_prices.set_index('PMPDKEY', inplace=True)
    banking_price = 50.0

    total_revenues_daily = pd.Series(0, index=df_data.index)

    for x, key in district_pmp_keys.items():
        direct_deliveries = pd.Series(0, index=df_data.index)
        use_bank = pd.Series(0, index=df_data.index)
        sell_bank = pd.Series(0, index=df_data.index)

        if x in district_prices.index:
            water_price = district_prices.loc[x, 'PMPWCST']
        else:
            continue

        # Custom rule for losthills
        if district_display_key == 'losthills':  #losthill has part of the district being accounted for in Wonderfull 
            output_key_lh = 'losthills_tableA_delivery'
            output_key_wf = 'wonderful_LHL_tableA_delivery'
            if output_key_lh in df_data and output_key_wf in df_data:
                direct_deliveries = df_data[output_key_lh] + df_data[output_key_wf] 
        
        # Deliveries made directly to customers
        else: 
            contract_list = ['tableA', 'cvpdelta', 'exchange', 'cvc', 'friant1', 'friant2', 'kings', 'kaweah', 'tule', 'kern']
            for contract in contract_list:
                for output in ['delivery', 'flood_irrigation', 'flood']:
                    output_key = f"{district_pmp_keys[x]}_{contract}_{output}"
                    if output_key in df_data:
                        direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)

        output_key = f"{key}_recharged"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.sub(df_data[output_key], fill_value=0)
            direct_deliveries = direct_deliveries.clip(lower=0)

        for output in ['inleiu_recharge', 'leiupumping']:
            output_key = f"{key}_{output}"
            if output_key in df_data:
                sell_bank = sell_bank.add(df_data[output_key], fill_value=0)

        output_key = f"{key}_inleiu_irrigation"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)
            sell_bank = sell_bank.add(df_data[output_key], fill_value=0)

        output_key = f"{key}_recover_banked"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)
            use_bank = use_bank.add(df_data[output_key], fill_value=0)

        output_key = f"{key}_exchanged_GW"
        if output_key in df_data:
            use_bank = use_bank.add(df_data[output_key], fill_value=0)

        output_key = f"{key}_exchanged_SW"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)

        if district_display_key == key:
            total_revenues_daily = (direct_deliveries * water_price + 
                                    sell_bank * banking_price - 
                                    use_bank * banking_price) / 1000.0

            return total_revenues_daily.resample(frequencies).last() #delivery is cumulative, so we take the last value of the year
#%%
def analyze_correlation_between_pumping_and_revenue(district_display_key, years_to_include):
    start_time = time.time() 
    results_folder = "results"
    district_pmp_keys = set_district_keys()
    # district_display_key = 'losthills'

    all_records = [] 

    for year in range(1996, 2025):
        for simulation in range(1, 101):
            folder_name = f"{simulation}_{year}"
            output_file = f"{results_folder}/{folder_name}/results.hdf5"

            if not os.path.exists(output_file):
                print(f"File {output_file} does not exist. Skipping...")
                continue

            datDaily = get_results_sensitivity_number_outside_model(output_file, '')

            # Adjust data to include only the requested years
            days_to_include = 365 * years_to_include
            datDaily = datDaily.iloc[:days_to_include]

            total_revenue = calculate_district_revenues(datDaily, district_display_key, district_pmp_keys)

            daily_pump_data = datDaily.loc[:, ['delta_HRO_pump', 'delta_TRP_pump']]
            if district_display_key == "semitropic":
                daily_pump_data['total_pump'] = daily_pump_data['delta_HRO_pump'] + daily_pump_data['delta_TRP_pump']
            else:
                daily_pump_data['total_pump'] = daily_pump_data['delta_HRO_pump']

            yearly_pump_data = daily_pump_data.resample('AS-OCT').sum()['total_pump']

            # total_revenue is a time series that can be handled similarly
            total_revenue = total_revenue.resample('AS-OCT').last().iloc[:years_to_include]

            aligned_data = pd.DataFrame({
                'Annual Revenue': total_revenue,
                'total_pump': yearly_pump_data
            }).dropna()

            for total_pump, annual_revenue in zip(aligned_data['total_pump'], aligned_data['Annual Revenue']):
                all_records.append({
                    'Year': year,
                    'Simulation': simulation,
                    'Total Pumping': total_pump,
                    'Annual Revenue': annual_revenue
                })

    df_to_save = pd.DataFrame(all_records)
    csv_filename = f"plots/pumping_vs_revenue_{district_display_key}_{years_to_include}yr.csv"
    df_to_save.to_csv(csv_filename, index=False)

    x = np.array(df_to_save['Total Pumping'])
    y = np.array(df_to_save['Annual Revenue'])

    # slope, intercept = np.polyfit(x, y, 1)
    # line = slope * x + intercept

    # plt.figure(figsize=(12, 6))
    # plt.scatter(x, y, alpha=0.6)
    # plt.plot(x, line, 'r-', label=f'Fit line: y={slope:.2f}x + {intercept:.2f}')
    # plt.title(f"{district_display_key}, {years_to_include}-Year")
    # plt.xlabel("Total Pumping")
    # plt.ylabel("Annual Revenue ($Million)")
    # plt.grid(False)

    # corr_coef = np.corrcoef(x, y)[0, 1]
    # r_squared = corr_coef ** 2
    # plt.annotate(f"Correlation: {r_squared:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
    #              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='b', facecolor='white'))

    # # plt.legend()
    # fig_filename = f"plots/basis_risk_{district_display_key}_{years_to_include}yr.png"
    # plt.savefig(fig_filename)
    # plt.show()

    # end_time = time.time()
    # print(f"Runtime: {end_time - start_time:.2f} seconds")

    # os.makedirs("plots", exist_ok=True)
    return x, y

# Change the parameter to 1 or 2 to analyze only the first year or first two years
# analyze_correlation_between_pumping_and_revenue(years_to_include=1)
# %%
