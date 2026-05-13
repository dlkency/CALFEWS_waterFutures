import numpy as np
import pandas as pd
import collections as cl
import sys
import calendar
import json
import matplotlib.pyplot as plt
import time
import h5py
from .util import *
from .reservoir_cy import Reservoir
from .delta_cy import Delta
from .canal_cy import Canal
from .district_cy import District
from .private_cy import Private
from .waterbank_cy import Waterbank
from .contract_cy import Contract
from .participant_cy import Participant
#from .model_cy import model
#from calfews_src.model_cy import model
#from calfews_src.model_cy import 

def initialization_routine(model, initial_condition):
  
  
  attribute_dict = {}
  delta = Delta(model, 'delta', 'DEL', model.model_mode)
  print(f"model.T = {model.T}, type = {type(model.T)}")
  print('columnsssssssssss')
  print(model.df[0].columns)
  #model.df[0].trt_inf
  attribute_dict['trinity'] = Reservoir(model, 'trinity', 'TRT', model.model_mode, initial_condition)
  #attribute_dict['lewiston'] = Reservoir(model, 'lewiston', 'SHA', model.model_mode, initial_condition)
  #attribute_dict['whiskeytown'] = Reservoir(model, 'whiskeytown', 'WSK', model.model_mode, initial_condition)  

  attribute_dict['delta']=Delta(model,'DEL',model.model_mode,initial_condition)
  
  #attribute_dict['keswick'] = Reservoir(model, 'keswick', 'SHA', model.model_mode, initial_condition)
  #attribute_dict['springcreek'] = Reservoir(model, 'springcreek', 'SHA', model.model_mode, initial_condition)
  #print(f"model.T = {model.T}, type = {type(model.T)}")
  #reservoir_list = [attribute_dict[key] for key in ['trinity','lewiston','whiskeytown']]
  
  attribute_dict['reservoir_list']=[attribute_dict[attr] for attr in ['trinity']]

  if model.model_mode == 'validation':

    for reservoir_obj in attribute_dict['reservoir_list']:
      reservoir_obj.find_release_func(model)

    
    for reservoir_obj in attribute_dict['reservoir_list']:
     reservoir_obj.create_flow_shapes(model)

  if model.model_mode == 'simulation':

    for reservoir_obj in attribute_dict['reservoir_list']:
      print("short_starting_year:", model.short_starting_year)
      print("short_ending_year:", model.short_ending_year)
      print("T_short:", model.T_short)
      print("df_short first/last:", model.df_short[0].index[0], model.df_short[0].index[-1])
     
      reservoir_obj.find_release_func(model)
      
    for reservoir_obj in attribute_dict['reservoir_list']:
     reservoir_obj.create_flow_shapes(model)

  return attribute_dict


def simulate_routine_trinity(model, t: int, self, shasta_fcr: float) -> tuple:
    
    delta_obj=model.delta


    print('SIMULATION RAN')
  
    d=model.day_year[t]
    da = model.day_month[t]
    dowy = model.dowy[t]
    m = model.month[t]
    y = model.year[t]
    
    year_index = y - model.starting_year


    calc_wytype_trinity(model,t)    

    for reservoir_obj in model.reservoir_list:
      
      reservoir_obj.release_environmental_trt1(t,d,m,dowy,y,model.first_d_of_month[year_index], delta_obj.forecastSTWYT)
      #PreRod
      #reservoir_obj.release_environmental_trt_PREROD(t,d,m,dowy,y,model.first_d_of_month[year_index], delta_obj.forecastSTWYT)
      
      print('passed through release environmental')
      print(reservoir_obj.key)
    
  
    
    model.trinity.compute_diversion(t,m,y, delta_obj.forecastSTI[t],model.trinity.eos_day,cfs_tafd,shasta_fcr)
    #Prerod
    #model.trinity.compute_diversion_PREROD(t,m,y, delta_obj.forecastSTI[t],model.trinity.eos_day,cfs_tafd,shasta_fcr)
    
    print('####passed in compute diversions####')
    print('Shasta FCR')
    print(shasta_fcr)
    print('Consumed Releases')
    print(reservoir_obj.consumed_releases)


 
    model.trinity.diversions[t] = model.trinity.consumed_releases #+ model.trinity.envmin- model.trinity.restoration[t]
    #model.trinity_diversions[t] = model.trinity.diversions[t]
   # model.trinity_div.append(model.trinity.diversions[t])
    #print('trt div in trt')
    #print(model.trinity_div)
 #   trinity_div = model.trinity.diversions[t]



    print('trt diversions in simulate')
    print(model.trinity.diversions[t])

    print('trt consumed releases in simulate')
    print(model.trinity.consumed_releases)

    print('trt envmin in simulate')
    print(model.trinity.envmin)

    print('trt restoration in simulate')
    print(model.trinity.restoration[t])

    daily_value=model.trinity.diversions[t]
    model.trinity.trt_consumed_daily.append(daily_value)
    print('DIVERSIONS DAILY IN TRT SYSTEM ')
    print(model.trinity.trt_consumed_daily)
    print(model.trinity.trt_consumed_daily[len(model.trinity.trt_consumed_daily)-1])

    if len(model.trinity.trt_consumed_daily) % 364 == 0:
             
        yearly_total=0.0
        yearly_total = sum(model.trinity.trt_consumed_daily)  
        model.trinity.trt_consumed_yearly.append(yearly_total)          
        print('DIVERSIONS TOTAL IN TRT SYSTEM')
        print(model.trinity.trt_consumed_yearly)
        print('MODEL YEAR IN TRT')
        print(y)
        index=len(model.trinity.trt_consumed_yearly)
        print(model.trinity.trt_consumed_yearly[len(model.trinity.trt_consumed_yearly)-1])
        print('index before error')
        print(index)
        print('TOTAL YEARLY FNF TRINITY')
       
        #print((self.yearly_totals[index-1]) * 1000)
        #print(self.yearly_totals[index-1])
        #yearly_diversion = (self.yearly_totals[index-1]*1000) - rest_flow
        model.trinity.trt_consumed_daily.clear()
        #self.daily_values_this_year = []
        #self.yearly_totals = []
        print('DAILY TRT SYSTEM 2')
        print(model.trinity.trt_consumed_daily)
        #self.yearly_totals.clear()
   
    for reservoir_obj in model.reservoir_list:
      reservoir_obj.find_available_storage(t,m,da,dowy)
      print('got here in find available storage')
      
    
    trinity_stored_release=model.trinity.envmin
    print('//////////envminnnnn/////////')
    print(trinity_stored_release)
  

    for reservoir_obj in model.reservoir_list:
      reservoir_obj.find_flow_pumping(t,m,dowy,year_index,model.days_in_month,model.dowy_eom,model.trinity.forecastWYT,'env')
      reservoir_obj.days_til_full[t]=min(reservoir_obj.numdays_fillup['env'],reservoir_obj.numdays_fillup['lookahead'])
   

    for reservoir_obj in model.reservoir_list:	
      print("[TRT OBJ CHECK] key=", reservoir_obj.key,
        "id=", id(reservoir_obj),
        "use_storage_override=", getattr(reservoir_obj, "use_storage_override", None),
        "S_obs is None?", getattr(reservoir_obj, "S_obs", None) is None,
        "override_last_t=", getattr(reservoir_obj, "override_last_t", None),
        "debug_storage_override=", getattr(reservoir_obj, "debug_storage_override", None))
      reservoir_obj.step_trt(t)
      print('PASSED IN STEP_TRT')


    #return flood_release_trinity,flood_volume_trinity
    return  model.trinity.diversions[t]




def calc_wytype_trinity(model, t: int):
 ####Water year type of trinity, determined by fnf downstream lewiston
 delta_obj=model.delta
 print('printing deltaaaaaaaaa')
 print(delta_obj.forecastSTI[t])


 print('NNNNN DELTA FORECAST STIII NNNNN')
 print(delta_obj.forecastSTI[t])
 print(t)

 if delta_obj.forecastSTI[t] <= 0.65:
   model.trinity.forecastWYT = "C"
   delta_obj.forecastSTWYT = "C"
 elif delta_obj.forecastSTI[t] <= 1.025:
    model.trinity.forecastWYT = "D"
    delta_obj.forecastSTWYT = "D"
 elif delta_obj.forecastSTI[t] <= 1.35:
    model.trinity.forecastWYT = "BN"
    delta_obj.forecastSTWYT = "BN"
 elif delta_obj.forecastSTI[t] <= 2:
   model.trinity.forecastWYT = "AN"
   delta_obj.forecastSTWYT = "AN"
 else:
    model.trinity.forecastWYT = "W"
    delta_obj.forecastSTWYT = "W"

 print('<<<<<DELTA FORECAST STWYT TYPE & VALUE >>>>>>')
 print(type(delta_obj.forecastSTWYT))
 print(delta_obj.forecastSTWYT)
 print('TRINITY FORECAST WYT')
 print(model.trinity.forecastWYT)

 return model.trinity.forecastWYT, delta_obj.forecastSTWYT
 
