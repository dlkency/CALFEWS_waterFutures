import numpy as np
import pandas as pd
import collections as cl
import sys
import calendar
import json
import matplotlib.pyplot as plt
import time
import h5py
from pandas.core.generic import T
from pyparsing import ParseSyntaxException
from .util import *
from .reservoir_cy import Reservoir
from .delta_cy import Delta
from .canal_cy import Canal
from .district_cy import District
from .private_cy import Private
from .waterbank_cy import Waterbank
from .contract_cy import Contract
from .participant_cy import Participant

class Metropolitan():
  def __init__(self):
    pass
    return

  def initialization_routine(self, model, initial_condition, scenario = 'baseline'):
    attribute_dict = {}

    ########################################################################################################################################
    # Southern California Water Districts - State Water Project Contractors
    ########################################################################################################################################

    attribute_dict['antelopekern'] = District(model, 'antelopekern', 'AVK', scenario)
    attribute_dict['crestline'] = District(model, 'crestline', 'CRL', scenario)
    attribute_dict['desert'] = District(model, 'desert', 'DST', scenario)
    attribute_dict['mojavewater'] = District(model, 'mojavewater', 'MJV', scenario)
    attribute_dict['palmdale'] = District(model, 'palmdale', 'PMD', scenario)
    attribute_dict['sangabriel'] = District(model, 'sangabriel', 'SGL', scenario)
    attribute_dict['santaclarita'] = District(model, 'santaclarita', 'SCL', scenario)
    attribute_dict['ventura'] = District(model, 'ventura', 'VTA', scenario)
    attribute_dict['coachellavalley'] = District(model, 'coachellavalley', 'CCHV', scenario)
    attribute_dict['sanbernadino'] = District(model, 'sanbernadino', 'SBN', scenario)
    attribute_dict['sangorgonio'] = District(model, 'sangorgonio', 'SGO', scenario)
    attribute_dict['metropolitan'] = District(model, 'metropolitan', 'MET', scenario)
    
    #########################################################################################################################################
    # Metropolitan System Constituent Water Districts - Post-Processing Distribution of 'MET' deliveries amongst these districts
    #########################################################################################################################################

    attribute_dict['anaheim'] = District(model, 'anaheim', 'ANA', scenario)
    attribute_dict['beverlyhills'] = District(model, 'beverlyhills', 'BVH', scenario)
    attribute_dict['burbank'] = District(model, 'burbank', 'BUR', scenario)
    attribute_dict['compton'] = District(model, 'compton', 'COM', scenario)
    attribute_dict['fullerton'] = District(model, 'fullerton', 'FUL', scenario)
    attribute_dict['glendale'] = District(model, 'glendale', 'GLN', scenario)
    attribute_dict['longbeach'] = District(model, 'longbeach', 'LNB', scenario)
    attribute_dict['losangeles'] = District(model, 'losangeles', 'LAX', scenario)
    attribute_dict['pasadena'] = District(model, 'pasadena', 'PAS', scenario)
    attribute_dict['sanfernando'] = District(model, 'sanfernando', 'SFN', scenario)
    attribute_dict['sanmarino'] = District(model, 'sanmarino', 'SMO', scenario)
    attribute_dict['santaana'] = District(model, 'santaana', 'SNA', scenario)
    attribute_dict['santamonica'] = District(model, 'santamonica', 'SNM', scenario)
    attribute_dict['torrance'] = District(model, 'torrance', 'TOR', scenario)
    attribute_dict['calleguas'] = District(model, 'calleguas', 'CLG', scenario)
    attribute_dict['centralbasin'] = District(model, 'centralbasin', 'CBN', scenario)
    attribute_dict['eastern'] = District(model, 'eastern', 'EAS', scenario)
    attribute_dict['foothill'] = District(model, 'foothill', 'FOT', scenario)
    attribute_dict['inlandempire'] = District(model, 'inlandempire', 'INE', scenario)
    attribute_dict['lasvirgenes'] = District(model, 'lasvirgenes', 'LAV', scenario)
    attribute_dict['orangecounty'] = District(model, 'orangecounty', 'ORA', scenario)
    attribute_dict['sandiegocounty'] = District(model, 'sandiegocounty', 'SDC', scenario)
    attribute_dict['threevalleys'] = District(model, 'threevalleys', '3VY', scenario)
    attribute_dict['westbasin'] = District(model, 'westbasin', 'WBN', scenario)
    attribute_dict['riversidecounty'] = District(model, 'riversidecounty', 'RSC', scenario)
    attribute_dict['uppersangabriel'] = District(model, 'uppersangabriel', 'USG', scenario)
    
    ###########################################################################################################################################
    #Metropolitan Water District Reservoirs - Included in the Storage Attribute of the District object for now - potential additions for later
    ###########################################################################################################################################

    #attribute_dict['diamondvalley'] = Reservoir(model, 'diamondvalley', 'DVL', model.model_mode, initial_condition) #Main Metropolitan Storage Reservoir
    #attribute_dict['mathews'] = Reservoir(model, 'mathews', 'MAT', model.model_mode, initial_condition) #End Point CRA if water doesn't go straight to Metropolitan Member Districts
    #attribute_dict['skinner'] = Reservoir(model, 'skinner', 'SKN', model.model_mode, initial_condition) #Treatment/Origin Deliveries to San Diego Aqueduct

    ###########################################################################################################################################
    #State Water Project (SWP) Reservoirs and others relevant to the Metropolitan System - Manage Storage on Non-Dummy Reservoirs 
    ###########################################################################################################################################

    #attribute_dict['castaiclake'] = Reservoir(model, 'castaiclake', 'CAST', model.model_mode, initial_condition)  #West Branch California Aqueduct
    #attribute_dict['silverwood'] = Reservoir(model, 'silverwood', 'SHA', model.model_mode, initial_condition)  #East Branch California Aqueduct
    #attribute_dict['perris'] = Reservoir(model, 'perris', 'PER', model.model_mode, initial_condition) #East Branch California Aqueduct
    attribute_dict['dummy'] = Reservoir(model, 'dummy', 'DUMMY', model.model_mode, initial_condition) #Colorado River Aqueduct Supply (Lake Havasu)
    attribute_dict['dummy2'] = Reservoir(model, 'dummy2', 'DUMMY2', model.model_mode, initial_condition) #Los Angeles Aqueduct Supply (Lake Mono/Owens Valley Watershed)
    attribute_dict['ecICS'] = Reservoir(model, 'ecICS', 'ICS', model.model_mode, initial_condition) #Metropolitan Lake Mead ICS Account - Only Used in colorado_pumping() to exchange water between itself and 'dummy'
    
    ############################################################################################################################################
    #New Aqueducts - Facilitate Reservoir/Contract/District system and define delivery capacity in JSON
    ############################################################################################################################################

    attribute_dict['swpdistribution'] = Canal('swpdistribution', 'SWPD', scenario) #Would be used to transfer water from Lake Perris if we set it up as a Reservoir Class object - Not Currently used in Bubble
    #attribute_dict['foothillfeeder'] = Canal('foothillfeeder', 'FTH', scenario) #Would be used to transfer water from Lake Castaic if we set it up as a Reservoir Class object
    #attribute_dict['sandiegoaqueduct'] = Canal('sandiegoaqueduct', 'SDA', scenario) #Would be used to transfer water from Lake Skinner to deliver to the San Diego County Water Authority - rate limiter
    attribute_dict['coloradoaqueduct'] = Canal('coloradoaqueduct', 'CRA', scenario)  #Colorado River Aqueduct - Connected to 'dummy'
    attribute_dict['losangelesaqueduct'] = Canal('losangelesaqueduct', 'LAA', scenario) #Los Angeles Aqueduct - Connected to 'dummy2'
    
    ############################################################################################################################################
    #Water Banks - Necessary to bring most into 'modelso' for interactions with San Luis Deliveries
    ############################################################################################################################################

    attribute_dict['westside'] = Waterbank(model, 'westside', 'WSD', scenario) #AVEK Water Bank - Operation Years in update_regulations_metro_south
    attribute_dict['eastside'] = Waterbank(model, 'eastside', 'ESD', scenario) #AVEK Water Bank - Operation Years in update_regulations_metro_south
    attribute_dict['amargosa'] = Waterbank(model, 'amargosa', 'UAG', scenario) #AVEK Water Bank - Operation Years in update_regulations_metro_south
    attribute_dict['highdesert'] = Waterbank(model, 'highdesert', 'HDT', scenario) # AVEK Water Bank Operational (2023-Present) excluded for now, need to update historical changes - initialized but capacity set to 0 in JSON

    #############################################################################################################################################
    #Contracts - Only New Contract Objects Defined Here, New Objects can be tied to existing contracts ('tableA') in 'modelso'
    #############################################################################################################################################

    attribute_dict['coloradocompact'] = Contract(model, 'coloradocompact', 'CMPT') #Colorado River Aqueduct Water - Max Contract Allocation for MET currently represents capacity of 'CRA'
    attribute_dict['owensvalley'] = Contract(model, 'owensvalley', 'OVY') #Los Angeles Aqueduct Water - Max Contract Allocation for MET currently represents capacity of 'LAA'
    attribute_dict['preferential'] = Contract(model, 'preferential', 'PRF') #Preferential Rights setup for buying water from the Metropolitan - would be annual changes to project_contract['preferential']
    #attribute_dict['secondary_wholesaler'] = Contract(model, 'secondary_wholesaler', 'SWS') #Second Bubbble Contract to run off of deliveries to Metropolitan - (Orange County Water District probably)

    #############################################################################################################################################
    
    self.further_initialization(model, attribute_dict) 
    
    return attribute_dict

  def further_initialization(self, model, attribute_dict):

    ############################################################################################################################
    #Completes initialization for objects initialized in attribute_dict, preps for crossover to the original modelso object
    ############################################################################################################################

    self.initialize_metro_reservoirs(model, attribute_dict)
    self.initialize_metro_districts(model, attribute_dict)
    self.initialize_metro_contracts(model, attribute_dict)
    self.initialize_metro_water_banks(model, attribute_dict)
    self.initialize_metro_canals(model, attribute_dict)
    self.dummy_canal_direction(attribute_dict)
  
    return

  ###########################################################################################
  #Setup and Remaining Initialization functions
  ###########################################################################################

  def initialize_metro_reservoirs(self, model, attribute_dict):

  ###########################################################################################
  #Reservoir Initialization - reservoir_list split between actual reservoirs and dummy reservoirs
  ###########################################################################################

    #reservoir_names = ['diamondvalley', 'mathews', 'skinner', 'castaiclake', 'perris'] 
    reservoir_names = []
    self.reservoir_list = [attribute_dict[name] for name in reservoir_names]

    dummy_reservoir_names = ['dummy', 'dummy2', 'ecICS']
    self.dummy_reservoir_list = [attribute_dict[name] for name in dummy_reservoir_names]

    if model.model_mode == 'climate_ensemble':
      for reservoir_obj in self.reservoir_list:
          reservoir_obj.find_release_func(model)
      for reservoir_obj in self.reservoir_list:
        reservoir_obj.create_flow_shapes(model)
    elif model.model_mode == 'validation':
      for reservoir_obj in self.reservoir_list:
        reservoir_obj.find_release_func(model)
      for reservoir_obj in self.reservoir_list:
        reservoir_obj.create_flow_shapes(model)
    else:
      for reservoir_obj in self.reservoir_list: 
        reservoir_obj.find_release_func(model)
      for reservoir_obj in self.reservoir_list:
        reservoir_obj.create_flow_shapes(model)

    #No Reservoirs (maybe Mono) that will have WYT dependent outflow releases but having the list exist may be beneficial for running any reservoir_cy.pyx functions
    expected_outflow_releases = {}
    for wyt in ['W', 'AN', 'BN', 'D', 'C']: #wyt - [wet, above normal, below normal, dry, critical]
      expected_outflow_releases[wyt] = np.zeros(366)
    inflow_list = self.reservoir_list 
    for reservoir_obj in inflow_list:
      reservoir_obj.downstream_short = [_ * cfs_tafd for _ in model.df_short[0]['%s_gains'% reservoir_obj.key].values]

    for reservoir_obj in inflow_list:
      reservoir_obj.calc_expected_min_release(model, expected_outflow_releases, np.zeros(12), 0)

    for reservoir_obj in self.dummy_reservoir_list:
      pass
    return

  def initialize_metro_districts(self, model, attribute_dict):
  ######################################################################################################
  #District Initialization 
  ######################################################################################################
    self.mwd_member_names = ['calleguas', 'centralbasin', 'eastern', 'foothill', 'inlandempire', 'uppersangabriel', 'riversidecounty', 'lasvirgenes', 
                        'orangecounty', 'threevalleys', 'westbasin', 'anaheim', 'beverlyhills', 'burbank', 'compton', 'fullerton', 'glendale', 
                       'longbeach', 'losangeles', 'pasadena', 'sanfernando', 'sanmarino', 'santaana', 'santamonica', 'torrance', 'sandiegocounty']
    self.mwd_member_list = [attribute_dict[name] for name in self.mwd_member_names]
    
    self.swp_contractor_names = ['antelopekern', 'crestline', 'desert', 'mojavewater', 'palmdale', 'sangabriel', 
                            'santaclarita', 'ventura', 'coachellavalley', 'sanbernadino', 'sangorgonio', 'metropolitan']
    self.swp_contractors = [attribute_dict[name] for name in self.swp_contractor_names ]

    ####################################################################################################
    #categorization of Metropolitan's (MWD) 26 water districts based on FY 2024-2025 Cost of Service document

    #municipal_water_districts_list = [attribute_dict[name] for name in ['calleguas', 'centralbasin', 'eastern', 'foothill', 'inlandempire', 'uppersangabriel', 
    #                                                                    'riversidecounty', 'lasvirgenes', 'orangecounty', 'threevalleys', 'westbasin']]
    #cities_list = [attribute_dict[name] for name in ['anaheim', 'beverlyhills', 'burbank', 'compton', 'fullerton', 'glendale', 'longbeach', 'losangeles', 
    #                                                  'pasadena', 'sanfernando', 'sanmarino', 'santaana', 'santamonica', 'torrance']]
    #CWA_list = [attribute_dict['sandiegocounty']]

    self.district_list = self.swp_contractors + self.mwd_member_list

    ####################################################################################################
    #have to setup a version of each list in the Metropolitan Class to run self.district_list calls in setup functions for pesitcide acreage, pmp model, and landiq acreage
    self.district_keys = {}
    self.district_keys_len = {}
 
    for district_obj in self.district_list:
      self.district_keys[district_obj.key] = district_obj
      self.district_keys_len[district_obj.key] = len(district_obj)

    #model.district_list.extend(self.district_list)
    #if model.demand_type == 'pesticide': #if avek ag added will need this
      #model.load_pesticide_acreage() #to use it this way district_list has to be an attribute of the model
    #elif model.demand_type == 'pmp':
      #model.load_pmp_model()
    #elif model.demand_type == 'landiq':
      #model.load_landiq_acreage()
    #model.allocate_private_contracts() #initializes .private_acreage, which if left as noneType triggers typeError in .find_baseline_demands

    #crop_life = 25
    for district_obj in self.district_list:
      district_obj.private_acreage = {}
      if district_obj.has_pesticide:
        district_obj.private_fraction = [0.0 for _ in range(self.number_years)]
        for crops in district_obj.acreage_by_year:
          district_obj.private_acreage[crops] = np.zeros(self.number_years)
      elif district_obj.has_pmp:
        for crops in district_obj.pmp_acreage:
          district_obj.private_acreage[crops] = 0.0
      else:
        district_obj.private_fraction = [0.0]
        for crops in district_obj.crop_list:
          district_obj.private_acreage[crops] = 0.0

    for district_obj in self.district_list:
      district_obj.find_baseline_demands(0, model.non_leap_year, model.days_in_month)
    
    return

  def initialize_metro_contracts(self, model, attribute_dict):
  ###########################################################################################
  #Contract Initialization - 'preferential', others need to 
  ###########################################################################################
    self.contract_names = ['coloradocompact', 'owensvalley', 'preferential']
    self.contract_list = [attribute_dict[name] for name in self.contract_names] 
    self.contract_keys = {}
    for contract_obj in self.contract_list:
      self.contract_keys[contract_obj.name] = contract_obj

    return

  def initialize_metro_water_banks(self, model, attribute_dict):
  ##################################################################################################################
  #Waterbank Initialization - Need to eventually account for where I can aggregate private storage to Metropolitan
  ##################################################################################################################
    self.waterbank_names = ['westside', 'eastside', 'amargosa', 'highdesert'] #remember to add a turnout to the 'calaqueduct.json' or appropriate_canal.json when adding a waterbank, add to canal_district
    self.waterbank_list = [attribute_dict[name] for name in self.waterbank_names]
    self.leiu_list = []
    for district_obj in self.district_list:
      if (district_obj.in_leiu_banking == 1):
        self.leiu_list.append(district_obj)
    
    for district_obj in self.district_list:
      district_obj.delivery_location_list = []
      district_obj.delivery_location_list.append(district_obj.key)
      district_obj.deliveries[district_obj.key + '_recharged'] = np.zeros(model.number_years)
      for waterbank_obj in self.waterbank_list:
        district_obj.deliveries[waterbank_obj.key + '_recharged'] = np.zeros(model.number_years)
        district_obj.delivery_location_list.append(waterbank_obj.key)
      for leiu_obj in self.leiu_list:
        district_obj.deliveries[leiu_obj.key + '_recharged'] = np.zeros(model.number_years)
        district_obj.delivery_location_list.append(leiu_obj.key)
    
    if model.model_mode == 'validation': ###placeholder
      attribute_dict['westside'].banked['AVK'] = 5
      attribute_dict['eastside'].banked['AVK'] = 5
      attribute_dict['amargosa'].banked['AVK'] = 5
      attribute_dict['amargosa'].banked['PMD'] = 5
      attribute_dict['highdesert'].banked['AVK'] = 5
      attribute_dict['highdesert'].banked['MET'] = 5
    elif model.model_mode == 'simulation': ###placeholder
      attribute_dict['westside'].banked['AVK'] = 5
      attribute_dict['eastside'].banked['AVK'] = 5
      attribute_dict['amargosa'].banked['AVK'] = 5
      attribute_dict['amargosa'].banked['PMD'] = 5
      attribute_dict['highdesert'].banked['AVK'] = 5
      attribute_dict['highdesert'].banked['MET'] = 5
    return

  def initialize_metro_canals(self, model, attribute_dict):
  ###################################################################################################################
  #Canal Initialization - New Canals (CRA, LAA) - May need a distribution canal where delivery rates can be limited
  ###################################################################################################################
    canal_names = ['coloradoaqueduct', 'losangelesaqueduct']
    self.canal_list = [attribute_dict[name] for name in canal_names]

    for district_obj in self.district_list:
      district_obj.infrastructure_shares = {}
    for canal in self.canal_list:
      try:
        for district_key in canal.ownership_shares:
          district_obj = self.district_keys[district_key]
          district_obj.infrastructure_shares[canal.name] = canal.ownership_shares[district_key]
      except:
        pass
    
    return
  
  def southmodel_metro_object_associations(self, model):
  ###############################################################################################################################################################################
  #### This function manages the replacement of the soCAL district from the original CALFEWS model with the collection of SWP districts being added from the Metropolitan Class
  ###############################################################################################################################################################################
    
    self.imported_canal_list = [model.calaqueduct] # Re-do 'calaqueduct' object associations with the new metropolitan objects
    
    idx_avek = self.swp_contractor_names.index('antelopekern')
    ordered_metro_swp_list = (self.swp_contractor_names[:idx_avek + 1] + self.waterbank_names + self.swp_contractor_names[idx_avek + 1:]) #find avek and insert wbs in the correct order of delivery moving from sanluisstate to MET
    ordered_metro_swp_objects = [getattr(model, name) for name in ordered_metro_swp_list]

    model.canal_district['caa'].extend(ordered_metro_swp_objects)
    model.canal_district_len['caa'] = len(model.canal_district['caa'])

    ### Set up aqueduct connections to MET, Dummy Reservoirs
    model.canal_district['cra'] = [getattr(model, name) for name in ['dummy', 'ecICS', 'metropolitan']]
    model.canal_district['laa'] = [getattr(model, name) for name in ['dummy2', 'metropolitan']] 
    model.canal_district['swpd'] = [getattr(model, name) for name in self.mwd_member_names]

    for key in ['cra', 'laa', 'swpd']:
      model.canal_district_len[key] = len(model.canal_district[key])

    for canal_obj in (self.imported_canal_list + self.canal_list):
      model.canal_by_name[canal_obj.name] = canal_obj
      canal_obj.num_sites = model.canal_district_len[canal_obj.name]
      canal_obj.turnout_use = [0.0 for _ in range(canal_obj.num_sites)]
      canal_obj.flow = [0.0 for _ in range(canal_obj.num_sites+1)]
      canal_obj.demand = {}
      canal_obj.turnout_frac = {}
      canal_obj.recovery_flow_frac = {}
      canal_obj.daily_flow = {}
      canal_obj.daily_turnout = {}
      for i in range(len(model.canal_district[canal_obj.name])):
        canal_obj.daily_flow[model.canal_district[canal_obj.name][i].key] = np.zeros(model.T)
        canal_obj.daily_turnout[model.canal_district[canal_obj.name][i].key] = np.zeros(model.T)
      for z in ['contractor', 'turnout', 'excess', 'priority', 'secondary', 'initial', 'supplemental']:
        canal_obj.demand[z] = np.zeros(canal_obj.num_sites)
        canal_obj.turnout_frac[z] = np.zeros(canal_obj.num_sites)
        canal_obj.recovery_flow_frac[z] = np.ones(canal_obj.num_sites)
      for canal_obj2 in [model.calaqueduct]: 
        canal_obj.demand[canal_obj2.name] = np.zeros(canal_obj.num_sites)
        canal_obj.turnout_frac[canal_obj2] = np.zeros(canal_obj.num_sites)
        canal_obj.recovery_flow_frac[canal_obj2] = np.ones(canal_obj.num_sites)

    model.canal_priority['cra'] = [model.coloradoaqueduct]
    model.canal_priority['laa'] = [model.losangelesaqueduct]

    model.reservoir_contract['DUMMY'] = [model.coloradocompact]
    model.reservoir_contract['ICS'] = [model.coloradocompact]
    model.reservoir_contract['DUMMY2'] = [model.owensvalley]

    for district_obj in self.district_list:
      district_obj.carryover_rights = {}
      for contract_obj in self.contract_list:
        if contract_obj.type == 'right':
          if district_obj.has_pesticide: #probably will be able to get rid of this for these - there are no rights in this part of the model
            district_obj.carryover_rights[contract_obj.name] = contract_obj.carryover*district_obj.rights[contract_obj.name]['carryover']*(1.0-district_obj.private_fraction[0])
          else:
            district_obj.carryover_rights[contract_obj.name] = contract_obj.carryover*district_obj.rights[contract_obj.name]['carryover']*(1.0-district_obj.private_fraction[0])
        else:
          district_obj.carryover_rights[contract_obj.name] = 0.0

    for district_obj in model.district_list:
      district_obj.reservoir_contract = {}
      for reservoir_obj in model.reservoir_list + self.dummy_reservoir_list:
        use_reservoir = 0
        for contract_obj in model.reservoir_contract[reservoir_obj.key]:
          for contract_key_dis in district_obj.contract_list:
            if contract_obj.name == contract_key_dis:
              use_reservoir = 1
              break
        if use_reservoir == 1:
          district_obj.reservoir_contract[reservoir_obj.key] = 1
        else:
          district_obj.reservoir_contract[reservoir_obj.key] = 0
    for private_obj in model.private_list:
      private_obj.reservoir_contract = {}
      for reservoir_obj in model.reservoir_list + self.dummy_reservoir_list:
        use_reservoir = 0
        for contract_obj in model.reservoir_contract[reservoir_obj.key]:
          for district_key in private_obj.district_list:
            district_obj = model.district_keys[district_key]
            for contract_key_dis in district_obj.contract_list:
              if contract_obj.name == contract_key_dis:
                use_reservoir = 1
                break
        if use_reservoir == 1:
          private_obj.reservoir_contract[reservoir_obj.key] = 1
        else:
          private_obj.reservoir_contract[reservoir_obj.key] = 0

    for private_obj in model.city_list:
      private_obj.reservoir_contract = {}
      for reservoir_obj in model.reservoir_list + self.dummy_reservoir_list:
        use_reservoir = 0
        for contract_obj in model.reservoir_contract[reservoir_obj.key]:
          for district_key in private_obj.district_list:
            district_obj = model.district_keys[district_key]
            for contract_key_dis in district_obj.contract_list:
              if contract_obj.name == contract_key_dis:
                use_reservoir = 1
                break
        if use_reservoir == 1:
          private_obj.reservoir_contract[reservoir_obj.key] = 1
        else:
          private_obj.reservoir_contract[reservoir_obj.key] = 0

    #Contract Reservoir should not be a list, single reservoir object
    model.contract_reservoir['CMPT'] = model.dummy 
    model.contract_reservoir['OVY'] = model.dummy2

    model.canal_contract['caa'] = model.canal_contract['caa'] + [model.coloradocompact, model.owensvalley]
    model.canal_contract['cra'] = [model.swpdelta, model.coloradocompact, model.owensvalley]
    model.canal_contract['laa'] = [model.swpdelta, model.coloradocompact, model.owensvalley]

    model.contract_turnouts['coloradocompact'] = ['cra']
    model.contract_turnouts['owensvalley'] = ['laa']

    model.reservoir_canal['DUMMY'] = [model.coloradoaqueduct]
    model.reservoir_canal['ICS'] = [model.coloradoaqueduct]
    model.reservoir_canal['DUMMY2'] = [model.losangelesaqueduct]

    model.canal_reservoir['cra'] = [model.dummy, model.ecICS]
    model.canal_reservoir['laa'] = [model.dummy2]
	
    return

  def object_list_extender(self, model):
  #########################################################################################################################
  ### This function transfers the necessary metropolitan objects into initialization functions within the southern model
  #########################################################################################################################

    wb_starter = [getattr(model, name) for name in self.waterbank_names]
    contracts_starter = [getattr(model, name) for name in self.contract_names]
    
    import_district_list = model.district_list
    district_extender = [getattr(model, name) for name in self.swp_contractor_names]

    for district_obj in self.district_list:
      model.district_keys[district_obj.key] = district_obj
      model.district_keys_len[district_obj.key] = len(district_obj)
    
    model.waterbank_list = wb_starter
    model.contract_list = contracts_starter
    model.district_list = import_district_list + district_extender

    return
  
  ###########################################################################################
  #Supply Additions for SWP Validation/Eventual CRA & LAA Modeling
  ###########################################################################################
  
  def met_imported_historical(self, model, t):
  #########################################################################################
  #Import Fixed Values for CRA and LAA Delivery to Metropolitan for SWP Validation
  #########################################################################################
    y = model.year[t]
    year_index = y - model.starting_year

    imported_supplies = pd.read_csv(r'C:/Users/evan/OneDrive\Documents/GitHub/CALFEWSv2/calfews_src/MET UWMP Supply Data_CALFEWS 1996_2024_format.csv')  #Historic Import Data MET UWMP
    imported_supplies['Colorado River Aqueduct'] = imported_supplies['Colorado River Aqueduct'] / 1000
    imported_supplies['LA Aqueduct'] = imported_supplies['LA Aqueduct'] / 1000
    imported_supplies['Sum Surface'] = imported_supplies['Sum Surface'] / 1000
    metropolitan_profile = [0.0643, 0.0607, 0.0673, 0.0758, 0.0857, 0.0974, 0.1069, 0.106, 0.0975, 0.0917, 0.0785, 0.0681]
    self.cra_wy_supplies, self.laa_wy_supplies = ([] for _ in range(2))

    
    for i in range(-3,9): #convert calendar year imports to water year
      if i < 0:
        self.cra_wy_supplies.append(imported_supplies['Colorado River Aqueduct'][year_index]*metropolitan_profile[i])
        self.laa_wy_supplies.append(imported_supplies['LA Aqueduct'][year_index]*metropolitan_profile[i])
      else:
        if year_index >= 28:
          self.cra_wy_supplies.append(imported_supplies['Colorado River Aqueduct'][year_index]*metropolitan_profile[i])
          self.laa_wy_supplies.append(imported_supplies['LA Aqueduct'][year_index]*metropolitan_profile[i])
        else:
          self.cra_wy_supplies.append(imported_supplies['Colorado River Aqueduct'][year_index+1]*metropolitan_profile[i])
          self.laa_wy_supplies.append(imported_supplies['LA Aqueduct'][year_index+1]*metropolitan_profile[i])
    self.cra_supplies = sum(self.cra_wy_supplies)
    self.laa_supplies = sum(self.laa_wy_supplies)
    return
  
  def met_assign_supply(self, reservoir_obj, t):
  #########################################################################################
  #Bring Fixed Values into modelso, runs at the end of the last day of the water year
  #########################################################################################
    if reservoir_obj.key == 'DUMMY':
      return_supply = self.cra_supplies
    else:
      return_supply = self.laa_supplies

    reservoir_obj.S[t] = return_supply

    return return_supply
  
  def update_metropolitan_mdd(self, model, wateryear, t):
  #########################################################################################
  #Assigns Metropolitan Annual Demand (MDD) to the aggregation of their observed deliveries
  #########################################################################################
    validation = pd.read_csv(r'C:/Users/evan/OneDrive\Documents/GitHub/CALFEWSv2/calfews_src/MET UWMP Supply Data_CALFEWS 1996_2024_format.csv')
    metropolitan_profile = [0.0643, 0.0607, 0.0673, 0.0758, 0.0857, 0.0974, 0.1069, 0.106, 0.0975, 0.0917, 0.0785, 0.0681]
    historical_swp_final_percent = [1,1,1,1,.9,.39,.7,.9,.65,.9,1,.6,.35,.4,.5,.8,.65,.35,.05,.2,.6,.85,.35,.75,.3,.05,.05,1,.4]
    recharge_names = ['antelopekern', 'crestline','coachellavalley', 'desert', 'mojavewater', 'sanbernadino', 'sangorgonio', 'ventura', 'palmdale', 'sangabriel', 'santaclarita']
    water_years, swp_wy_supplies, cra_wy_supplies, laa_wy_supplies, swp_wy_sum, cra_wy_sum, laa_wy_sum, total_wy = ([] for _ in range(8))

    #convert calendar year imports to water year
    for ii in range(1996, 2025):
        for i in range(-3,9):
            if i < 0:
                swp_wy_supplies.append(validation['State Water Project'][ii-1996]*metropolitan_profile[i])
                cra_wy_supplies.append(validation['Colorado River Aqueduct'][ii-1996]*metropolitan_profile[i])
                laa_wy_supplies.append(validation['LA Aqueduct'][ii-1996]*metropolitan_profile[i])
            else:
                if ii >= 2024:
                    swp_wy_supplies.append(validation['State Water Project'][ii-1996]*metropolitan_profile[i])
                    cra_wy_supplies.append(validation['Colorado River Aqueduct'][ii-1996]*metropolitan_profile[i])
                    laa_wy_supplies.append(validation['LA Aqueduct'][ii-1996]*metropolitan_profile[i])
                else:
                    swp_wy_supplies.append(validation['State Water Project'][ii-1996+1]*metropolitan_profile[i])
                    cra_wy_supplies.append(validation['Colorado River Aqueduct'][ii-1996+1]*metropolitan_profile[i])
                    laa_wy_supplies.append(validation['LA Aqueduct'][ii-1996+1]*metropolitan_profile[i])
            if i == 8:
                water_years.append(ii)
                swp_wy_sum.append(sum(swp_wy_supplies))
                cra_wy_sum.append(sum(cra_wy_supplies))
                laa_wy_sum.append(sum(laa_wy_supplies))
                total_wy.append(sum(swp_wy_supplies)+sum(cra_wy_supplies)+sum(laa_wy_supplies))
                swp_wy_supplies.clear()
                cra_wy_supplies.clear()
                laa_wy_supplies.clear()

    data = [water_years, swp_wy_sum, cra_wy_sum, laa_wy_sum, total_wy]
    column_names = ['Water Year', 'State Water Project', 'Colorado River Aqueduct', 'LA Aqueduct', 'Sum Surface']
    df_dict = dict(zip(column_names, data))
    validation = pd.DataFrame(df_dict)
    validation['Water Year'] = validation['Water Year'].astype(str)
    validation['Water Year'] = pd.to_datetime(validation['Water Year'], format='%Y') + pd.DateOffset(months=-3)
    validation.set_index('Water Year', inplace=True)

    for i in range(1996, 2025):
      if wateryear == i-1996:
        if i == 1996:
          model.metropolitan.MDD = 1*(validation['Sum Surface'][wateryear]/1000)
        else:
          model.metropolitan.MDD = 1*(validation['Sum Surface'][wateryear + 1]/1000)

    for district_obj in [getattr(model, name) for name in recharge_names]:
      for i in range(1996, 2024):
        if wateryear == i-1996:
          if wateryear == 0:
            district_obj.MDD = 1*4170.0*district_obj.project_contract['tableA']*historical_swp_final_percent[i-1996]
          else:
            district_obj.MDD = 1*4170.0*district_obj.project_contract['tableA']*historical_swp_final_percent[i-1995]
        if i != 2005 and i != 1998:  
          if district_obj.name in ['santaclarita','antelopekern', 'coachellavalley']:
            district_obj.MDD = max(district_obj.MDD, 30)
            if district_obj.MDD >= 25:
              district_obj.must_fill = 1
            else:
              district_obj.must_fill = 0
          if district_obj.name in ['sanbernadino']:
            district_obj.MDD = max(district_obj.MDD, 60)
            if district_obj.MDD >= 60:
              district_obj.must_fill = 1
            else:
              district_obj.must_fill = 0
          elif district_obj.name == 'crestline':
            district_obj.MDD = max(district_obj.MDD, 1)
            if district_obj.MDD == 1:
              district_obj.must_fill = 1
            else:
              district_obj.must_fill = 0
          else:
            district_obj.MDD = max(district_obj.MDD, 5)
            if district_obj.MDD >= 5:
              district_obj.must_fill = 1
            else:
              district_obj.must_fill = 0
    return

  def colorado_pumping(self, model, t, wyt):
  ############################################################################################################################
  #Colorado Aqueduct System - Incorporate Metropolitan ICS Account, Compact Allocations, Exchange with Coachella Valley, etc.
  ############################################################################################################################
    
    dowy = model.dowy[t]
    if model.year[t] == model.starting_year:
      self.oct_ICS_bool = False
      self.withdraw_bool = False

    #####################################################################################################################################################
    #Import Lake Mead Water Level on Jan. 1 of the Water Year from the October, update it on Actual Jan. 1 - https://www.usbr.gov/uc/water/hydrodata/
    #####################################################################################################################################################
    
    inputdata = pd.read_csv(r'C:/Users/evan/OneDrive\Documents/GitHub/CALFEWSv2/calfews_src/Mead_Elv_CALFEWS.csv')
    #if dowy >= 0:
      #lakemead_level_oct = inputdata.iloc[model.year[t]-1996][0] #this is set up as actual october 1st rn, I want it to be the USBR projection of jan 1 water level from oct 1
    #if dowy >= 92:
      #lakemead_level_jan = inputdata.iloc[model.year[t]-1996][1] ###commented for test run

    if dowy >= 0:
      lakemead_level_jan = inputdata.iloc[model.year[t]-1996][1]

    ###########################################################################################################################
    # Assign Ruleset based on Validation Year or Simulation, determine legally required system conservation through DCP
    ###########################################################################################################################

    if model.model_mode == 'validation':
      if model.year[t] < 2003:
        ruleset = 'sevenparties'
      elif model.year[t] < 2019:
        ruleset = 'QSA'
      elif model.year[t] <= 2026: #Lower Basin Drought Contingency Plan introduces first curtailments for California, 2026 end of regulation for now but can be adjusted depending on new operating plan
        ruleset = 'lbDCP'
      elif model.year[t] <= 2026:
        ruleset = 'IRA' #system conservation reduction adjustments from federal funding to USBR via the Inflation Reduction Act
      else:
        ruleset = 'lbDCP'
    else:
      ruleset = 'lbDCP'  #THIS IS WHERE WE WOULD WRITE OUT ANY ALTERNATIVE rulesets for simulation mode, set to current operating rules for now

    if ruleset == 'lbDCP' or ruleset == 'IRA': ###will have to build in some planning variable so that the model has some idea of what to expect based on the October Storage
      if lakemead_level_jan < 1045: #my understanding is that nearly the entirety of CA's DCP responsibility is currently held by Metropolitan, hence these values
        if lakemead_level_jan > 1040:
          dcp_contribution = 200
        elif lakemead_level_jan > 1035:
          dcp_contribution = 250
        elif lakemead_level_jan > 1030:
          dcp_contribution = 300
        elif lakemead_level_jan > 1025:
          dcp_contribution = 350
        else:
          dcp_contribution = 350
      else:
        dcp_contribution = 0
    else:
      dcp_contribution = 0

    ############################################################################################################################################################
    # Unused Seven-Party Agreement 1/2/3b. (Palo Verde Irrigation District, Yuma Project) < 420,000 AF
    ############################################################################################################################################################

    pr1_cutoff = 420 #QSA defined
    if ruleset != 'sevenparties':
      pr1_consumptive = 0 #will need to build these out, 0 rn for testing
      pr1_conservation = 0 #will need to build these out, 0 rn for testing
      if dowy == 180:
        if wyt == 'W' or 'AN':
          pr1_consumptive = 415 #heavy assumptions
          pr1_conservation = 0
        elif wyt == 'BN':
          pr1_consumptive = 435
          pr1_conservation = 0
        elif wyt == 'D':
          pr1_consumptive = 460 
          pr1_conservation = 0
        elif wyt == 'C':
          pr1_consumptive = 500 
          pr1_conservation = 0

      pr1_usage = pr1_consumptive + pr1_conservation
      pr1_adjustment = pr1_cutoff - pr1_usage
      if dowy != 180:
        pr1_adjustment = 0 ###PLACEHOLDER####

    ############################################################################################################################################################
    #Unused Seven-Party Agreeement 3a. (IID, CVWD)  < (3430 - System Conservation - PPF) ###this whole section will need some work
    ############################################################################################################################################################
      
    pr3a_cutoff = 3430
    if ruleset != 'sevenparties':
      pr3a_consumptive = 0 #will need to build these out, 0 for testing
      pr3a_conservation = 0 #will need to build these out, 0 for testing
      if dowy == 180:
        if wyt == 'W' or 'AN':
          pr3a_consumptive = 3425 #heavy assumptions
          pr3a_conservation = 0
        elif wyt == 'BN':
          pr3a_consumptive = 3450
          pr3a_conservation = 0
        elif wyt == 'D':
          pr3a_consumptive = 3500 
          pr3a_conservation = 0
        elif wyt == 'C':
          pr3a_consumptive = 3550 
          pr3a_conservation = 0
      pr3a_usage = pr3a_consumptive + pr3a_conservation
      pr3a_adjustment = pr3a_cutoff - pr3a_usage
      if dowy != 180:
        pr3a_adjustment = 0 ####PLACEHOLDER#####

    ############################################################################################################################################################
    #Assign Water Transfers to MET/SDCWA from other seven-party agreement partners
    ############################################################################################################################################################
    
    SDCWA_exchange = 0
    sanluisrey_exchange = 0
    SNWA_storage_exchange = 0
    self.oct_ICS_adjustment = 150
    quechan_forbearance = 6.5
    needles_lowerCOsupply = 0
    IID_All_American_Savings = 105
    if model.year[t] <= 2023:
      PVID_fallowing = 70

    if model.model_mode == 'validation':
      if model.year[t] == 2004:
        SDCWA_exchange = 10
        SNWA_storage_exchange = 10
      elif model.year[t] == 2005:
        SDCWA_exchange = 20
        SNWA_storage_exchange = 10
      elif model.year[t] == 2006:
        SDCWA_exchange = 30
        SNWA_storage_exchange = 5
      elif model.year[t] == 2007:
        SDCWA_exchange = 40
        needles_lowerCOsupply = 5
      elif model.year[t] == 2008:
        SDCWA_exchange = 50
        sanluisrey_exchange = 2
        SNWA_storage_exchange = 45
        needles_lowerCOsupply = 6
      elif model.year[t] == 2009:
        SDCWA_exchange = 140.2
        sanluisrey_exchange = 5
        SNWA_storage_exchange = 27.5
        needles_lowerCOsupply = 2.35
      elif model.year[t] == 2010:
        SDCWA_exchange = 151.5
        sanluisrey_exchange = 16
        SNWA_storage_exchange = 8.16
        needles_lowerCOsupply = 3.5
      elif model.year[t] == 2011:
        SDCWA_exchange = 159.9
        needles_lowerCOsupply = 3.5
      elif model.year[t] == 2012:
        SDCWA_exchange = 186.9
        SNWA_storage_exchange = 62
        needles_lowerCOsupply = 3.5
      elif model.year[t] in range(2013,2018):
        SDCWA_exchange = 177.7
        needles_lowerCOsupply = 6.5 
        if model.year[t] == 2013:
          needles_lowerCOsupply = 4.5 
          SNWA_storage_exchange = 75
        elif model.year[t] == 2014:
          SNWA_storage_exchange = 65
        elif model.year[t] == 2015:
          SNWA_storage_exchange = 150
      elif model.year[t] == 2018:
        SDCWA_exchange = 207.7
      elif model.year[t] == 2019:
        SDCWA_exchange = 237.7
      elif model.year[t] == 2020:
        SDCWA_exchange = 270.2
      elif model.year[t] == 2021:
        SDCWA_exchange = 270.2
      elif model.year[t] in range(2022,2024):
        SDCWA_exchange = 277.7 #Full Contractual Obligation, Timeframe set by the QSA, Contractual Salton Sea Mitigation Measures Completed
      elif model.year[t] >= 2024:
        SDCWA_exchange = 227.7 #Inflation Reduction Act - System Reductions of 50,000 AFY for preserving Lake Mead Level

      if model.year[t] >= 2010:
        sanluisrey_exchange = 16
      if model.year[t] >= 2018:
        needles_lowerCOsupply = 9.5
    else:
      SDCWA_exchange = 227.7
      sanluisrey_exchange = 16
      SNWA_storage_exchange = 0

    if dowy == 0 and ruleset == 'sevenparties':
      model.metropolitan.projected_supply['coloradocompact'] = 1200
    elif dowy == 0:
      model.metropolitan.projected_supply['coloradocompact'] = 550 ###model.metropolitan.project_contract['coloradocompact']*model.coloradocompact.total #### BASE PRIORITY 4 = 550 check on this later
      model.metropolitan.projected_supply['coloradocompact'] -= dcp_contribution #assess any dcp system conservation based on lake mead level onto the forecasted supply for metropolitan
      if model.year[t] >= 1996:
        model.metropolitan.projected_supply['coloradocompact'] += IID_All_American_Savings
      if model.year[t] >= 2005:
        model.metropolitan.projected_supply['coloradocompact'] += PVID_fallowing
        model.metropolitan.projected_supply['coloradocompact'] += SDCWA_exchange
      if model.year[t] >= 2007:
        model.metropolitan.projected_supply['coloradocompact'] += needles_lowerCOsupply
        model.metropolitan.projected_supply['coloradocompact'] += quechan_forbearance
      if model.year[t] >= 2008:
        model.metropolitan.projected_supply['coloradocompact'] += SNWA_storage_exchange
        model.metropolitan.projected_supply['coloradocompact'] += sanluisrey_exchange
      if model.year[t] in range(2016,2019) or range(2020, 2025):
        bard_fallowing = 4 ###will have to find a way to get some variance in here too.
        model.metropolitan.projected_supply['coloradocompact'] += bard_fallowing

      if wyt == 'W' and model.year[t] >= 2006:
        self.oct_ICS_bool = True
        model.metropolitan.projected_supply['coloradocompact'] -= self.oct_ICS_adjustment
      
    ################################################################################################
    # ICS Usage
    ################################################################################################

    if model.year[t] >= 2006:
      CA_EC_ICS_delivery_limit = 400
      if model.year[t] >= 2010: ###example, build out with programs
        eligible_storage_capacity = 200
      else:
        eligible_storage_capacity = 100
      CA_EC_ICS_annual_storage_limit = max(400, eligible_storage_capacity)
      CA_EC_ICS_capacity = 1700

    if model.year[t] >= 2017:
      CA_EC_ICS_annual_storage_limit = 450

    if dowy == 180 and ruleset != 'sevenparties' and model.year[t] >= 2006:
      annual_demand = (.62)*model.metropolitan.MDD  #.5488
      model.metropolitan.projected_supply['coloradocompact'] += pr1_adjustment
      model.metropolitan.projected_supply['coloradocompact'] += pr3a_adjustment
      pre_ICS_supply = model.metropolitan.projected_supply['tableA'] + model.metropolitan.projected_supply['owensvalley'] + model.metropolitan.projected_supply['coloradocompact'] - 200 #adjustment for laa uncertainty
      if self.oct_ICS_bool == True:
        pre_ICS_supply += self.oct_ICS_adjustment
        self.oct_ICS_bool = False
      if model.year[t] > 2006:
        if annual_demand > pre_ICS_supply and lakemead_level_jan > 1025 and model.ecICS.S[t] != 0:
          self.withdraw_bool = True
          withdraw_request = annual_demand - pre_ICS_supply
          if withdraw_request <= CA_EC_ICS_delivery_limit and model.ecICS.S[t] >= withdraw_request:
            model.metropolitan.projected_supply['coloradocompact'] += withdraw_request
            model.ecICS.S[t] -= withdraw_request
          elif withdraw_request > CA_EC_ICS_delivery_limit and model.ecICS.S[t] >= withdraw_request:
            model.metropolitan.projected_supply['coloradocompact'] += CA_EC_ICS_delivery_limit
            model.ecICS.S[t] -= CA_EC_ICS_delivery_limit
          elif withdraw_request > model.ecICS.S[t]:
            model.metropolitan.projected_supply['coloradocompact'] += model.ecICS.S[t]
            model.ecICS.S[t] = 0
          else:  
            self.withdraw_bool = False
        else:
          self.withdraw_bool = False
      else:
        self.withdraw_bool = False
    #else:
      #withdraw_bool = False
    

    #################################################################################################
    #ICS Adjustments 
    #################################################################################################

    if dowy == 364 and self.withdraw_bool == False and model.year[t] >= 2006: ###is dowy an array we're iterating through here, for last day of the water year could you do something like dowy[-1]?
      available_capacity = max(CA_EC_ICS_capacity - model.ecICS.S[t], 0)
      deposit_volume = model.metropolitan.projected_supply['coloradocompact']
      #print('Available Capacity: ' + str(available_capacity), end = ' ')
      #print('Deposit Volume/EOY projected supply: ' + str(deposit_volume))
      if deposit_volume <= CA_EC_ICS_annual_storage_limit and available_capacity >= deposit_volume:
        model.ecICS.S[t] += deposit_volume
      elif deposit_volume <= CA_EC_ICS_annual_storage_limit and deposit_volume >= available_capacity:
        model.ecICS.S[t] += available_capacity
      elif deposit_volume >= CA_EC_ICS_annual_storage_limit:
        model.ecICS.S[t] += min(available_capacity, CA_EC_ICS_delivery_limit)
      else: 
        pass

    #################################################################################################
    #WWRF deliveries are important to correctly adjust CRA down to MET deliveries
    #################################################################################################

    ### TEMPORARY HARDCODE FOR THESE ADJUSTMENTS TO TEST THE REST OF THE CODE
    if dowy == 0:
      CVWD_df = pd.read_csv(r'C:/Users/evan/OneDrive\Documents/GitHub/CALFEWSv2/calfews_src/CVWD_CRAval.csv') ####WRRF deliveries/total adjustment from CRA Diversions to Supply Available to Service Area + SDCWA
      CVWD_adjustment = CVWD_df['Non Service Area Supply'][model.year[t] - 1995] #this has not been adjusted for water years yet, may have to do the ol' (3/4)x+(1/4)y aggregation --> actually use average CRA diversion profile to reaggregate
      model.metropolitan.projected_supply['coloradocompact'] -= CVWD_adjustment

    ####This will need to be replaced with 6 yr aggregation, IID obligations, TableA transfer setups eventually, but need to get the rest operating smoothly before we can start testing that

    #################################################################################################
    #Assign Projected Supply to the Dummy Reservoir
    #################################################################################################

    if dowy == 364:
      self.withdraw_bool = False

    model.dummy.S[t] = model.metropolitan.projected_supply['coloradocompact'] #assign volume of approved water to be delivered to the dummy reservoir

    return 
  
  def mono_regulation(self, model, t):
  ############################################################################################################################################################################################
  #Mono Lake has been controlled by Environmental Regulations since 1989/1994 official decision - classifications based on Average Lake Elevation for the previous Runoff Year (April-March)
  #Variable Part of the Owens Valley Supply which feeds the Los Angeles Aqueduct
  ############################################################################################################################################################################################
    if model.model_mode == 'validation':
      if model.year[t] in [1996, 2015, 2016, 2017, 2023]: #selected years from analysis period for now to determine what the wa
        yearly_diversion = 4.5 #acre-ft
      else:
        yearly_diversion = 16
    else:
      yearly_diversion = 16

    return yearly_diversion

  def owens_valley_pumping(self, model, t, wyt):
  ###########################################################################################################################
  #Majority of City Rights through pre-1914 riparian land rights, now limited for dust control measures in the Owens Valley
  #Combines with smaller Mono Basin diversions to supply the LA Aqueduct
  ###########################################################################################################################
    if model.year[t] == model.starting_year:
      self.oct_laa_estimate = 0

    dowy = model.dowy[t]
    
    mono_basin_diversions = self.mono_regulation(model, t)
    if model.model_mode == 'validation':
      if wyt == 'W':
        if model.water_year[t] in range(0,3):
          owens_valley_runoff = 440.0
        else:
          owens_valley_runoff = 365.6
      elif wyt == 'AN':
        owens_valley_runoff = 240.0
      elif wyt == 'BN':
        owens_valley_runoff = 200.0
      elif wyt == 'D':
        owens_valley_runoff = 160.0
      else:
        if model.water_year[t] < 14:
          owens_valley_runoff = 135.0
        else:
          owens_valley_runoff = 64.0
    else:
      if wyt == 'W':
        owens_valley_runoff = 246.0
      elif wyt == 'AN' or 'BN':
        owens_valley_runoff = 192.0
      elif wyt == 'D':
        owens_valley_runoff = 143.0
      else:
        owens_valley_runoff = 71.4
    laa_supply = mono_basin_diversions +  owens_valley_runoff
    if dowy == 0:
      self.oct_laa_estimate = laa_supply
      model.metropolitan.projected_supply['owensvalley'] = laa_supply - 15
      model.dummy2.S[t] = model.metropolitan.projected_supply['owensvalley']
    elif dowy == 180:
      if laa_supply != self.oct_laa_estimate:
        difference = laa_supply - self.oct_laa_estimate
        model.metropolitan.projected_supply['owensvalley'] += (1/2)*difference
        model.owensvalley.available_water[t] += (1/2)*difference
        model.dummy2.S[t] += (1/2)*difference
    if dowy == 180:
      self.oct_laa_estimate = 0
    return
  
  ###########################################################################################
  #Dummy Reservoir Setup - South Model Dependencies
  ###########################################################################################

  def dummy_mandatory_release(self, model, reservoir_obj, m, year_index):
    release = 'demand'

    reservoir_obj.min_daily_uncontrolled = 0.0 ### rate at which flow has to be released in order to avoid overtopping
    reservoir_obj.max_daily_uncontrolled = 999.99 ### rate which flow cannot be released pass in order to avoid missing EOS targets
    reservoir_obj.uncontrolled_available = 0.0 ### maximum volume 'above' flood control, w/o releases
    reservoir_obj.numdays_fillup[release] = 999.99 ### number of days until reservoir fills
    reservoir_obj.numdays_fillup['lookahead'] = 999.99
    if reservoir_obj.name == 'dummy':
      reservoir_obj.min_daily_uncontrolled = (self.cra_supplies*model.metropolitan.urban_profile[m-1])/model.days_in_month[year_index][m-1]
      reservoir_obj.max_daily_uncontrolled = reservoir_obj.min_daily_uncontrolled
    else:
      reservoir_obj.min_daily_uncontrolled = (self.laa_supplies*model.metropolitan.urban_profile[m-1])/model.days_in_month[year_index][m-1]
      reservoir_obj.max_daily_uncontrolled = reservoir_obj.min_daily_uncontrolled
    return
  
  def dummy_canal_direction(self, attribute_dict):
    attribute_dict['coloradoaqueduct'].flow_directions['recharge']['cra'] = "normal"
    attribute_dict['coloradoaqueduct'].flow_directions['recovery']['cra'] = "normal"
    attribute_dict['losangelesaqueduct'].flow_directions['recharge']['laa'] = "normal"
    attribute_dict['losangelesaqueduct'].flow_directions['recovery']['laa'] = "normal"
    return
  
  ###########################################################################################
  #Future Metropolitan Modeling - Simulation + Internal Objects Handshake
  ###########################################################################################

  def test_metropolitan_internal_objects(self, model, t):
    #preferential contract allocation should be set equal to model.metropolitan.projected_supply['tableA'] + model.metropolitan.projected_supply['coloradocompact'], set project contract's off of that, makes similar adjustments as it approaches zero
    #some storage variable that serves as a secondary supply indicator with the allocation: diamondvalley.S[t] + castaic.S[t] + skinner.S[t] + silverwood.S[t] + otherres.S[t] etc.
    #should test creating another bubble for districts who buy recharged groundwater from a secondary wholesaler who contracts with Metropolitan (Orange County Water District) - will likely require another contract wrap
    
    d = model.day_year[t]
    da = model.day_month[t]
    dowy = model.dowy[t]
    m = model.month[t]
    y = model.year[t]
    wateryear = model.water_year[t]
    year_index = y - model.starting_year

    if y == model.starting_year and dowy == 0:
      self.reservoir_storage = 0 
    for district_obj in self.mwd_member_list:
      test_demand = 1800*district_obj.project_contract['preferential']
      district_obj.annualdemand[0] = test_demand
      district_obj.projected_supply['preferential'] = (model.metropolitan.projected_supply['tableA'] + model.metropolitan.projected_supply['coloradocompact'])*district_obj.project_contract['preferential']
      district_obj.deliveries['preferential'][wateryear] = (model.metropolitan.deliveries['tableA'][wateryear] + model.metropolitan.deliveries['coloradocompact'][wateryear])*district_obj.project_contract['preferential']
      if district_obj.name == 'losangeles':
        district_obj.projected_supply['preferential'] += model.metropolitan.projected_supply['owensvalley']
        district_obj.deliveries['preferential'][wateryear] += model.metropolitan.deliveries['owensvalley'][wateryear]
      #if district_obj.name == 'losangeles':
        #print('dowy: ' + str(dowy) + ' annual demand: ' + str(district_obj.annualdemand[0]))
        #print('dowy: ' + str(dowy) + ' deliveries: ' + str(district_obj.deliveries['preferential'][wateryear]))
      district_obj.annualdemand[0] = max(district_obj.annualdemand[0] - district_obj.deliveries['preferential'][wateryear], 0.0)
      
      if m == 9 and da == 30:
        test_demand = district_obj.deliveries['preferential'][wateryear]
        if district_obj.deliveries['preferential'][wateryear] > test_demand:
          district_obj.carryover = district_obj.deliveries['preferential'][wateryear] - district_obj.annualdemand[0]
          self.reservoir_storage += district_obj.carryover
          district_obj.carryover = 0

      district_obj.accounting_full(t, wateryear)
    return

  def simulate_metro(self, model, attribute_dict,t, cvp_alloc, swp_alloc):
  
  ###########################################################################################
  #To Be Run from main_cy.pyx file
  ###########################################################################################
    ####Maintain the same date/time accounting as the southern part of the model
    d = model.day_year[t]
    da = model.day_month[t]
    dowy = model.dowy[t]
    m = model.month[t]
    y = model.year[t]
    wateryear = model.water_year[t]
    year_index = y - model.starting_year
    if m == 12:
      m1 = 1
    else:
      m1 = m + 1

    if model.model_mode == 'validation':
      self.set_regulations_historical_metro(model, attribute_dict)
    else:
      self.set_regulations_current_metro(model, attribute_dict)
      
    return
  
  def preferential_rights_allocation(self, model, attribute_dict):
    #Metropolitan Water District Act Section 135 defines preferential rights for the member agencies of Metropolitan: Based on Property Tax and past capital contributions to Metropolitan, not Water Purchases
    #This will be allocated on a second pass through the Metropolitan System, water will be delivered to SWCs with TableA allocations first, then the Metropolitan Allocation will be distributed to meet demand within these bounds

    #if (y not in leap_years) or m!=2:
      #if dowy == 1: 
        #colorado_flow_shape = []
        #la_flow_shape = []
        #for i in len(range(0,365)):
          #colorado_flow_shape.append((metropolitan_profile[m]/month_lengths[m-1])*imported_supplies['Colorado River Aqueduct'][year_index])
          #la_flow_shape.append((metropolitan_profile[m]/month_lengths[m-1])*imported_supplies['LA Aqueduct'][year_index])
      #cra_supplies = (imported_supplies['Colorado River Aqueduct'][year_index]*metropolitan_profile[m])/month_lengths[m-1]
      #laa_supplies = (imported_supplies['LA Aqueduct'][year_index]*metropolitan_profile[m])/month_lengths[m-1]
      #if m == 9 and da == 31:
        #colorado_flow_shape.clear()
        #la_flow_shape.clear()
    #else:
      #cra_supplies = (imported_supplies['Colorado River Aqueduct'][year_index]*metropolitan_profile[m])/29
      #laa_supplies = (imported_supplies['LA Aqueduct'][year_index]*metropolitan_profile[m])/29



    #if reservoir_obj.key == 'DUMMY':
      #if (y not in leap_years) or m!=2:
        #reservoir_obj.S[t] = (metropolitan_profile[m]/month_lengths[m-1])*imported_supplies['Colorado River Aqueduct'][year_index]
      #else:
        #reservoir_obj.S[t] = (metropolitan_profile[m]/29)*imported_supplies['Colorado River Aqueduct'][year_index]
    #else:
      #if (y not in leap_years) or m!=2:
        #reservoir_obj.S[t] = (metropolitan_profile[m]/month_lengths[m-1])*imported_supplies['LA Aqueduct'][year_index]
      #else:
        #reservoir_obj.S[t] = (metropolitan_profile[m]/29)*imported_supplies['LA Aqueduct'][year_index]

    return
  
  ###########################################################################################
  #Historic Regulations changes - to be coded later, just leave comments for now
  ###########################################################################################

  def set_regulations_historical_metro(self, model, attribute_dict):
    if model.starting_year <= 2000:
      #attribute_dict['diamondvalley'].capacity = 0 #.tocs, .capacity, .carryover target, etc. attributes for the reservoir object
      pass
    elif model.starting_year >= 2005:
      pass
    return

  def set_regulations_current_metro(self, model, attribute_dict):
    if model.starting_year >= 2000:
      pass
    elif model.starting_year >= 2005:
      pass
    #2023 - agreements to reduce CRA usage to 2026 for creation of Intentionally Created Surplus supplies in Lake Mead
    #pre-2017 Arvid-Edison storage program - find start date, restarted 2021 with Friant Division SWP supplies
    #antelope valley change happening 2025 -> east kern storage program transitioning to East Kern High Desert Water Bank Progran
    #Bard fallowing 1.9 AF/acre saved <3000 acres -> lake mead

    #Brock Reservoir contribution -> 100,000 AF in Lake Mead Storage Account
    return

  def update_regulations_metro(self, model, attribute_dict):
    pass
    #USED FOR CHANGES WHICH HAPPEN DURING 1996-2016 calibration period, should only run as part of the simulate metro function when in validation mode

    #2002 - no unused allotment of CRA water available for first time - continued in perpetuity, all CRA deliveries to MET < 1.25 MAF of California's 4.4 MAF share
    #2003 - Quantification settlement agreement (QSA) between MET, CVWD, and IID - among these addition of 96,000 acre-ft from All-American Canal
    #2004 - SNWA release agreement
    #2006 - 20,000 AF settlement over water rights to the Quechan Tribe, MET pays for 7000 AF of this water until 2035
    #2007 - Intentionally Created Surplus agreement with IID
    #2011 - recieved  24,937 AF in exchange for participation in Yuma Desal Pilot
    #2013 - 23,750 AF from CR conservation pilot stored in Lake Mead ICS account to this day
    #2016 - change in distribution of the savings from AAC, set to 105,000/yr between Coachella Valley WD and Metropolitan
    
    return
