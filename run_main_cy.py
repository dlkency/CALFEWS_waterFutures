import sys
import os
import shutil
import numpy as np
import pandas as pd
from configobj import ConfigObj
from distutils.util import strtobool
from datetime import datetime
import main_cy  # bring in the main_cy class

start_time = datetime.now()

results_folder = sys.argv[1]  ### folder directory to store results, relative to base calfews directory
redo_init = int(sys.argv[2])   ### this should be 0 if we want to use saved initialized model, else 1
run_sim = int(sys.argv[3])   ### this should be 1 if we want to run sim, else 0 to just do init
initial_condition = sys.argv[4] ###this it the argument to pass to the start date of the reservoirs
init_location = sys.argv[5]   ### This is where the temporary initialization file is saved
print(initial_condition, type(initial_condition))

config = ConfigObj(f'{init_location}/runtime_params.ini')
print(config, type(config), 'Exist? ', bool(config))
cluster_mode = bool(strtobool(config['cluster_mode']))
scratch_dir = config['scratch_dir']

if cluster_mode:
  results_folder = scratch_dir + results_folder + '/'
else:
  results_base = 'results/' + results_folder + '/'
print('Results folder:', results_folder)
### if initialized main_cy object given, load it in
save_init = results_folder + '/main_cy_init.pkl'

### create results directory or remove old results
try:
  os.mkdir(results_folder)
except:
  try:
    os.remove(results_folder + '/results.hdf5')
  except:
    pass

if redo_init == 0:
  try:
    main_cy_obj = pd.read_pickle(save_init)
  except:
    redo_init = 1

### else start new initialization routine
if redo_init == 1:
  ### setup/initialize model
  print('#######################################################')
  print('Begin initialization...')
  sys.stdout.flush()

  try:
    os.mkdir(results_folder)  
  except:
    pass
  
  ### setup new model
  main_cy_obj = main_cy.main_cy(results_folder, runtime_file=f'{init_location}/runtime_params.ini')
  print(main_cy_obj.results_folder)
  a = main_cy_obj.initialize_py(initial_condition)  # add the intital_condition 

  if a == 0:
    ### save initialized model
    # pd.to_pickle(main_cy_obj, save_init)
    print('Initialization complete, ', datetime.now() - start_time)
    sys.stdout.flush()
  else:
    run_sim = 0

if run_sim == 1:
  ### main simulation loop
  a = main_cy_obj.run_sim_py(start_time) 
  print ('Simulation complete', datetime.now() - start_time)
  sys.stdout.flush()

  if a == 0:
    ### calculate objectives
    main_cy_obj.calc_objectives()
    print ('Objective calculation complete,', datetime.now() - start_time)

    ### output results
    main_cy_obj.output_results()
    print ('Data output complete,', datetime.now() - start_time)
    sys.stdout.flush()


