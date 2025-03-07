#
# Description: This script calculates the financial data for each district in CALFEWS model output.

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
#%%
sns.set_style('darkgrid')

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

 
# %%
save_folder = 'calfews_src/postprocess/'

def calculate_district_revenues(district_display_key, district_pmp_keys):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Assume daily data in each DataFrame indexed by DateTime
    # Read price data from PMP
    district_prices = pd.read_csv('calfews_src/postprocess/district_water_prices.csv')
    district_prices.set_index('PMPDKEY', inplace=True)
    banking_price = 50.0  # Volumetric rate to deliver water to a bank

    # Create dictionary to store daily district revenues
    district_revenues_daily = {}

    # Read CALFEWS output data
    output_file = 'results/1_2024/results.hdf5'
    df_data = get_results_sensitivity_number_outside_model(output_file, '')

    for x in district_pmp_keys:
        # Initialize arrays for daily calculations
        direct_deliveries = pd.Series(0, index=default_date_range)
        use_bank = pd.Series(0, index=default_date_range)
        sell_bank = pd.Series(0, index=default_date_range)

        # Find district's price to their customers
        if x in district_prices.index:
            water_price = district_prices.loc[x, 'PMPWCST']
            
        else:
            continue  # Skip if the key not found

        # Custom rule for losthills
        if district_display_key == 'losthills':  #losthill has part of the district being accounted for in Wonderfull 
            output_key_lh = 'losthills_tableA_delivery'
            output_key_wf = 'wonderful_LHL_tableA_delivery'
            # output_key_bel = 'belridge_tableA_delivery'
            # output_key_wbel = 'wonderful_BLR_tableA_delivery'
            # output_key_bdm = 'berrenda_tableA_delivery'
            # output_key_wbdm = 'wonderful_BDM_tableA_delivery'
            if output_key_lh in df_data and output_key_wf in df_data:
                direct_deliveries = df_data[output_key_lh] + df_data[output_key_wf] 
                # + df_data[output_key_bel] + df_data[output_key_wbel] + df_data[output_key_bdm] + df_data[output_key_wbdm]
        
        # Deliveries made directly to customers
        else: 
            contract_list = ['tableA', 'cvpdelta', 'exchange', 'cvc', 'friant1', 'friant2', 'kings', 'kaweah', 'tule', 'kern']
            for contract in contract_list:
                for output in ['delivery', 'flood_irrigation', 'flood']:
                    output_key = f"{district_pmp_keys[x]}_{contract}_{output}"
                    if output_key in df_data:
                        direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)


        #Recharged water is not delivered to customers (but it is counted in above deliveries), so no revenue is generated
        output_key = f"{district_pmp_keys[x]}_recharged"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.sub(df_data[output_key], fill_value=0)
            direct_deliveries = direct_deliveries.clip(lower=0)  # Ensure no negative deliveries

        # Banking transactions:  Deliveries/withdrawals from district bank by banking partners (revenue generating)
        for output in ['inleiu_recharge', 'leiupumping']:
            output_key = f"{district_pmp_keys[x]}_{output}"
            if output_key in df_data:
                sell_bank = sell_bank.add(df_data[output_key], fill_value=0)

        #Deliveries to district bank by banking partners (revenue generating) PLUS irrigation sale to in-district customer
        output_key = f"{district_pmp_keys[x]}_inleiu_irrigation"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)
            sell_bank = sell_bank.add(df_data[output_key], fill_value=0)

        #Withdrawals from out-of-district bank by district (cost generating), delivered to customers (revenue generating)
        output_key = f"{district_pmp_keys[x]}_recover_banked"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)
            use_bank = use_bank.add(df_data[output_key], fill_value=0)
    
        #Withdrawals from out-of-district bank by district (cost generating), exchanged with another district (revenue is already counted under contract delivery)
        output_key = f"{district_pmp_keys[x]}_exchanged_GW"
        if output_key in df_data:
            use_bank = use_bank.add(df_data[output_key], fill_value=0)

        #Recovered groundwater from another district delivered to district customers (revenue generating) that was exchanged for district surface water
        output_key = f"{district_pmp_keys[x]}_exchanged_SW"
        if output_key in df_data:
            direct_deliveries = direct_deliveries.add(df_data[output_key], fill_value=0)

        if district_display_key == district_pmp_keys[x]:
            # total_revenues_daily = (direct_deliveries * water_price ) / 1000.0
            total_revenues_daily = (direct_deliveries * water_price + sell_bank * banking_price - use_bank * banking_price) / 1000.0

            total_revenues_annually = total_revenues_daily.resample('AS-OCT').last() #resample to yearly revenue
            # print(water_price)
            print(direct_deliveries * water_price)
            print(sell_bank * banking_price)
            print(use_bank * banking_price)
            # print(use_bank)
            plt.figure(figsize=(12, 6))
            plt.plot(total_revenues_daily.index, total_revenues_daily.values, label="Daily Revenue")
            plt.title(f"Total Daily Revenue for {district_display_key}")
            plt.ylabel("Revenue in Million $")
            plt.xlabel("Date")
            plt.grid(True)
            plt.legend()
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.plot(total_revenues_annually.index, total_revenues_annually.values, label="Annual Revenue", color='orange')
            plt.title(f"Total Annual Revenue for {district_display_key} (Water Year)")
            plt.ylabel("Revenue in Million $")
            plt.xlabel("Water Year Start")
            plt.grid(True)
            plt.legend()
            plt.show()

            daily_filename = os.path.join(save_folder, f"{district_display_key}_daily_revenue_syn_1_2024.csv")
            annual_filename = os.path.join(save_folder, f"{district_display_key}_annual_revenue_syn_1_2024.csv")

            # Save to CSV
            total_revenues_daily.to_csv(daily_filename, header=True)
            total_revenues_annually.to_csv(annual_filename, header=True)


# Usage
district_pmp_keys = set_district_keys()
district_display_key = 'losthills'
default_date_range = pd.date_range(start='1924-10-01', end='2027-9-30', freq='D')
calculate_district_revenues(district_display_key, district_pmp_keys)
# %%
