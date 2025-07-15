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
  district_pmp_keys['D01'] = 'belridge'#add
  district_pmp_keys['D05'] = 'berrenda'#add
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
    direct_delivery_revenue = pd.Series(0, index=df_data.index)
    sell_bank_revenue = pd.Series(0, index=df_data.index)
    use_bank_revenue = pd.Series(0, index=df_data.index)

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

        elif district_display_key == 'belridge':  #losthill has part of the district being accounted for in Wonderfull 
            output_key_belridge = 'belridge_tableA_delivery'
            output_key_won_belridge = 'wonderful_BLR_tableA_delivery'
            direct_deliveries = df_data[output_key_belridge] + df_data[output_key_won_belridge] 

        elif district_display_key == 'berrenda':  #losthill has part of the district being accounted for in Wonderfull 
            output_key_berrenda = 'berrenda_tableA_delivery'
            output_key_won_berrenda = 'wonderful_BDM_tableA_delivery'
            direct_deliveries = df_data[output_key_berrenda] + df_data[output_key_won_berrenda] 

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

            return total_revenues_daily, direct_deliveries * water_price/ 1000.0, sell_bank * banking_price/ 1000.0, use_bank * banking_price / 1000.0
#%%
def analyze_correlation_between_pumping_and_revenue(district_display_key, years_to_include):
    start_time = time.time() 
    results_folder = "results/startyear_4_1"
    district_pmp_keys = set_district_keys()
    # district_display_key = 'losthills'

    all_records = [] 

    for year in range(1996, 2025):
        for simulation in range(1, 101):
            folder_name = f"{year}_{simulation}"
            output_file = f"{results_folder}/{folder_name}/results.hdf5"

            if not os.path.exists(output_file):
                print(f"File {output_file} does not exist. Skipping...")
                continue

            datDaily = get_results_sensitivity_number_outside_model(output_file, '')

            # Adjust data to include only the requested years
            days_to_include = 365 * years_to_include
            datDaily = datDaily.iloc[:days_to_include]

            total_revenue, delivery_rev, sellbank_rev, use_bank_rev = calculate_district_revenues(datDaily, district_display_key, district_pmp_keys)

            daily_pump_data = datDaily.loc[:, ['delta_HRO_pump']]
            yearly_pump_data = daily_pump_data.resample('AS-OCT').sum()['delta_HRO_pump']
            total_revenue = total_revenue.resample('AS-OCT').last().iloc[:years_to_include]
            delivery_rev = delivery_rev.resample('AS-OCT').last().iloc[:years_to_include]
            sellbank_rev = sellbank_rev.resample('AS-OCT').last().iloc[:years_to_include]
            use_bank_rev = use_bank_rev.resample('AS-OCT').last().iloc[:years_to_include]

            aligned_data = pd.DataFrame({
                'Annual Revenue': total_revenue,
                'Delivery Revenue': delivery_rev,
                'Sell Bank Revenue': sellbank_rev,
                'Use Bank Revenue': use_bank_rev,
                'total_pump': yearly_pump_data
            }).dropna()

            for total_pump, annual_revenue, delivery_revenue, sell_bank_revenue, use_bank_revenue in zip(aligned_data['total_pump'], aligned_data['Annual Revenue'], aligned_data['Delivery Revenue'], aligned_data['Sell Bank Revenue'], aligned_data['Use Bank Revenue']):
                all_records.append({
                    'Year': year,
                    'Simulation': simulation,
                    'Total Pumping': total_pump,
                    'Annual Revenue': annual_revenue,
                    'Delivery Revenue': delivery_rev,
                    'Sell Bank Revenue': sellbank_rev,
                    'Use Bank Revenue': use_bank_rev
                })

    df_to_save = pd.DataFrame(all_records)
    csv_filename = f"plots/pumping_vs_revenue_{district_display_key}_{years_to_include}yr.csv"
    df_to_save.to_csv(csv_filename, index=False)

    x = np.array(df_to_save['Total Pumping'])
    y = np.array(df_to_save['Annual Revenue'])

    return x, y
#%%
def analyze_correlation_between_pumping_and_revenue_historical(district_display_keys, years_to_include):

    results_folder = "results/"
    district_pmp_keys = set_district_keys()
    
    output_file = f"{results_folder}/110_year/results.hdf5"

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"File {output_file} does not exist.")

    datDaily = get_results_sensitivity_number_outside_model(output_file, '')
    
    days_to_include = 365 * years_to_include
    datDaily = datDaily.iloc[:days_to_include]

    # Initialize the total revenue across all districts
    total_revenue_across_districts = pd.Series(0, index=pd.date_range(start=datDaily.index[0], periods=years_to_include, freq='AS-OCT'))
    
    for district_display_key in district_display_keys:
        total_revenue, delivery_rev, sellbank_rev, use_bank_rev = calculate_district_revenues(
            datDaily, district_display_key, district_pmp_keys
        )

        # Aggregate revenue
        total_revenue_across_districts += total_revenue.resample('AS-OCT').last().iloc[:years_to_include]

    # Extract the relevant pumping data
    daily_pump_data = datDaily.loc[:, ['delta_HRO_pump']]
    yearly_pump_data = daily_pump_data.resample('AS-OCT').sum()['delta_HRO_pump']
    yearly_pump_data.index = total_revenue_across_districts.index

    # Create a dataframe with total annual revenue and total pumping
    aligned_data = pd.DataFrame({
        'Annual Revenue': total_revenue_across_districts,
        'Total Pumping': yearly_pump_data
    }).dropna()
    
    # Save the aggregated dataframe to a CSV file
    csv_filename = f"plots/pumping_vs_revenue_KCWA_110yr.csv"
    aligned_data.to_csv(csv_filename, index=False)

    # Return the data as numpy arrays for any necessary analysis
    x = np.array(aligned_data['Total Pumping'])
    y = np.array(aligned_data['Annual Revenue'])

    return x, y

# %%
def analyze_correlation_between_total_revenue_and_pumping():

    # start_time = time.time()
    results_folder = "results/startyear_4_1"
    district_pmp_keys = set_district_keys()

    districts = [
        'kerndelta', 'wheeler', 'westkern', 'belridge',
        'berrenda', 'semitropic', 'rosedale', 'buenavista',
        'cawelo', 'henrymiller', 'losthills'
    ]

    all_records = []

    for year in range(1996, 2025):
        for simulation in range(1, 101):
            folder_name = f"{year}_{simulation}"
            output_file = f"{results_folder}/{folder_name}/results.hdf5"

            if not os.path.exists(output_file):
                print(f"File {output_file} does not exist. Skipping...")
                continue

            datDaily = get_results_sensitivity_number_outside_model(output_file, '')

            # Initialize total revenue series
            total_revenue_all = pd.Series(0.0, index=datDaily.index)

            for district in districts:
                try:
                    district_revenue, _, _, _ = calculate_district_revenues(datDaily, district, district_pmp_keys)
                    total_revenue_all = total_revenue_all.add(district_revenue, fill_value=0)
                except Exception as e:
                    print(f"Error in district {district} for {year}_{simulation}: {e}")
                    continue

            # Resample annually (based on water year starting in October)
            annual_revenue = total_revenue_all.resample('AS-OCT').sum()
            total_pumping = datDaily['delta_HRO_pump'].resample('AS-OCT').sum()

            aligned_df = pd.DataFrame({
                'Year': annual_revenue.index.year,
                'Simulation': simulation,
                'Total Pumping': total_pumping.values,
                'Annual Revenue': annual_revenue.values
            })

            all_records.extend(aligned_df.to_dict(orient='records'))

    result_df = pd.DataFrame(all_records)
    result_df.to_csv("plots/pumping_vs_revenue_KCWA_30years.csv", index=False)

    x = result_df['Total Pumping'].values
    y = result_df['Annual Revenue'].values

    return x, y

# %%
