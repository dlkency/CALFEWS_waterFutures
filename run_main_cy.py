import sys
import os

import os, sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from configobj import ConfigObj
from distutils.util import strtobool
from datetime import datetime
import main_cy  


##SANITY, flush = true
print(" # RUN_MAIN FILE -> __file__ =", __file__, flush=True)
print(" ?? RUN_MAIN FILE -> cwd =", os.getcwd(), flush=True)
print(" $% RUN_MAIN FILE -?? argv =", sys.argv, flush=True)

#write a marker file into the results_folder if provided
try:
    rf = sys.argv[1] if len(sys.argv) > 1 else None
    if rf:
        Path(rf).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(rf, "RUN_MAIN_WHICH_FILE.txt")).write_text(
            f"__file__={__file__}\n"
            f"cwd={os.getcwd()}\n"
            f"argv={sys.argv}\n"
        )
except Exception as e:
    print("RUN_MAIN_CY FILE marker write failed:", e, flush=True)


#Read CSV, align actual storage with model's date index, store it 
#sentinel marker for missing storage in input file
def attach_storage_overrides(main_cy_obj,excel_path,cutoff_date,date_col="datetime",colmap=None, sentinel=-1.0, which="no", debug=True):

    if cutoff_date is None:
        print('SKIPPED ATTACH STORAGE OVERRIDE',flush=True)
        return -1

    # pick model
    if which == "no":
        m = main_cy_obj.modelno
    elif which == "so":
        m = main_cy_obj.modelso
    elif which =="trinity":
        m=main_cy_obj.trinity
    else:
        raise ValueError("which must be 'no', 'so' or 'trinity'")

    # default cmapping csv columns to model
    if colmap is None:
        colmap = {"SHA": "SHA_storage","ORO": "ORO_storage","FOL": "FOL_storage","NML": "NML_storage","TRT": "TRT_storage","SLS": "SLS_storage","SLF": "SLF_storage","SL":"SL_storage"}

    #model dates/horizon
    dates = pd.DatetimeIndex(m.df[0].index).normalize()
    T = int(m.T)

    #read obs 
    obs = pd.read_csv(excel_path)
    obs.columns = obs.columns.str.strip()

    #sanity
    if date_col not in obs.columns:
        raise KeyError(f"[OVERRIDE STORAGE] date_col='{date_col}' not found. Columns={list(obs.columns)}")

    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce").dt.normalize()
    obs = obs.dropna(subset=[date_col]).set_index(date_col).sort_index()

    #cutoff until override_last_t
    cutoff = pd.to_datetime(cutoff_date).normalize()
    idx = np.where(dates.to_numpy() <= cutoff.to_datetime64())[0]
    if len(idx):
        override_last_t=int(idx.max())
    else:
        override_last_t = -1
    
    #override_last_t = int(idx.max()) if len(idx) else -1
    #delete in the future
    if debug:
        last_date = (m.df[0].index[override_last_t] if override_last_t >= 0 else None)
        print(
            "[OVERRIDE STORAGE] cutoff_date =", cutoff,
            "| override_last_t =", override_last_t,
            "| override_last_date =", last_date
        )
    #align obsevred dates with model's and convert to taf
    def aligned_array(col):
        s = obs[col].reindex(dates)
        arr = s.to_numpy(dtype=float) / 1000.0  # AF to tAF
        arr = np.where(np.isfinite(arr), arr, sentinel)
        if len(arr) != T:
            raise ValueError(f"[OVERRIDE STORAGE] len(arr)={len(arr)} != T={T} for col={col}")
        return arr

    #delete in the future
    #only for printing, does not affect override math
    def set_res_debug(r):
        try:
            r.debug_storage_override = 1
            r.debug_storage_until_t = override_last_t
            r.debug_storage_every = 1
        except Exception:
            pass

    #find reservoir's key and attach override to a single reservoir object 
    #bec trinity and san luis not attachinggg
    def attach_one(r):
        key = str(getattr(r, "key", "")).strip().upper()
        if key not in colmap:
            return False

        col = colmap[key]
        if col not in obs.columns:
            raise KeyError(f"[OVERRIDE STORAGE] CSV missing column '{col}' for reservoir key={key}")

        r.S_obs = aligned_array(col)
        r.override_last_t = override_last_t
        r.use_storage_override = True
        set_res_debug(r)

        if debug:
            print(f"[OVERRIDE STORAGE] Attached {key}: col={col} | day0_obs(TAF)={r.S_obs[0]}")
        return True


    #Attach overrides to all reservoirs in reservoir_list 
    attached_keys = set()
    n_attached_reslist = 0

    for r in (getattr(m, "reservoir_list", []) or []):
        key = str(getattr(r, "key", "")).strip().upper()
        if attach_one(r):
            attached_keys.add(key)
            n_attached_reslist += 1

    if debug:
        print(f"[OVERRIDE STORAGE] Attached overrides via reservoir_list: {n_attached_reslist}")

#attach Trinity from a trt_list living in trinity model

    n_attached_trtlist = 0

# debug candidate places where trt_list might live
    candidates = [
        ("m.trt_list", getattr(m, "trt_list", None)),
        ("main_cy_obj.trinity.trt_list", getattr(getattr(main_cy_obj, "trinity", None), "trt_list", None)),
        ("m.trinity.trt_list", getattr(getattr(m, "trinity", None), "trt_list", None)),
    ]
    #find where trt list is
    trt_list = None
    trt_list_src = None
    for src, lst in candidates:
        if lst is not None:
            trt_list = lst
            trt_list_src = src
            break

    if trt_list is None:
        if debug:
            print("[OVR WARN] Could not find trt_list on m or trinity (checked: " +
                  ", ".join([c[0] for c in candidates]) + ")")
    else:
        for r in (trt_list or []):
            key = str(getattr(r, "key", "")).strip().upper()
            if key != "TRT":
                continue

        # attach exactly like reservoir_list
            if attach_one(r):
                n_attached_trtlist += 1
                if debug:
                    print(f"[OVERRIDE STORAGE] Attached overrides via {trt_list_src}: {n_attached_trtlist}")
            break

        if debug and n_attached_trtlist == 0:
            print(f"[OVR WARN] trt_list found at {trt_list_src} but no key=='TRT' item attached")


    return override_last_t


start_time = datetime.now()

results_folder = sys.argv[1]  ### folder directory to store results, relative to base calfews directory
redo_init = int(sys.argv[2])   ### this should be 0 if we want to use saved initialized model, else 1
run_sim = int(sys.argv[3])   ### this should be 1 if we want to run sim, else 0 to just do init
initial_condition = sys.argv[4] ###this it the argument to pass to the start date of the reservoirs
#init_location = sys.argv[5]   ### This is where the temporary initialization file is saved
print(initial_condition, type(initial_condition))


cutoff_date = None
init_location = None

# arg5 init_location if it looks like a folder containing runtime_params.ini, else cutoff_date 
if len(sys.argv) > 5:
    arg5 = sys.argv[5]
    ini5 = Path(arg5) / "runtime_params.ini"
    if ini5.exists():
        init_location = arg5
    else:
        cutoff_date = arg5

# arg6 cutoff_date (explicit) 
if len(sys.argv) > 6:
    cutoff_date = sys.argv[6]

# load config exactly like before (cluster vs local) 
if init_location is not None:
    config = ConfigObj(f"{init_location}/runtime_params.ini")
else:
    config = ConfigObj("runtime_params.ini")

print(config, type(config), "Exist? ", bool(config))


print('Results folder:', results_folder)
### if initialized main_cy object given, load it in
save_init = results_folder + '/main_cy_init.pkl'

###create results directory or remove old results
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
    print('PASSED HERE INIT')
    print(save_init)
    print(main_cy_obj)
  except:
    redo_init = 1
    print('PASSED HERE INIT2')

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
  #if len(sys.argv) > 4:
  if init_location is not None:
   print("init_location =", init_location, "| cutoff_date =", cutoff_date)

   main_cy_obj = main_cy.main_cy(results_folder, runtime_file=f'{init_location}/runtime_params.ini')
  else:
   print("init_location =", init_location, "| cutoff_date =", cutoff_date)

   main_cy_obj = main_cy.main_cy(results_folder)

  print(main_cy_obj.results_folder)


  a = main_cy_obj.initialize_py(initial_condition)
  print('THIS IS A')
  print(a)


  if a == 0:
    #CREATE OVERRIDE FILE IN RESUTLS FOLDER DEBUGGG
    Path(os.path.join(results_folder, "HIT_OVERRIDE_BLOCK.txt")).write_text("entered override block\n")
    print(" RUN_MAIN ABOUT TO CALL attach_storage_overrides ", flush=True)
    
    #use desired path for storage
    override_last_t = attach_storage_overrides(main_cy_obj,excel_path=r"CHANGE THE PATH TO YOUR DESIRED FILE", cutoff_date=cutoff_date, date_col="datetime", colmap={"SHA": "SHA_storage","ORO": "ORO_storage","FOL": "FOL_storage","TRT": "TRT_storage","NML":"NML_storage"}, which="no")

    override_last_t_trt = attach_storage_overrides(main_cy_obj,excel_path=r"CHANGE THE PATH TO YOUR DESIRED FILE",cutoff_date=cutoff_date,date_col="datetime",colmap={"TRT": "TRT_storage"},which="trinity")

    ##debug print
    tr_model = main_cy_obj.trinity
    for r in tr_model.reservoir_list:
        print("[TR MODEL RES]", r.key,
          "use_storage_override=", getattr(r,"use_storage_override",None),
          "S_obs None?", getattr(r,"S_obs",None) is None,
          "override_last_t=", getattr(r,"override_last_t",None),
          "debug_storage_override=", getattr(r,"debug_storage_override",None))


    override_last_so = attach_storage_overrides(main_cy_obj, excel_path=r"CHANGE THE PATH TO YOUR DESIRED FILE", cutoff_date=cutoff_date, date_col="datetime", colmap={"SLS": "SLS_storage","SLF": "SLF_storage","SNL":"SL_storage"},which="so")

    so_model = main_cy_obj.modelso


    # FORCE attach + debug for San Luis objects that exist in modelso 
    so = main_cy_obj.modelso
    ##debug printttttt
    for attr in ["sanluis", "sanluisstate", "sanluisfederal"]:
        r = getattr(so, attr, None)
        if r is None:
            print(f"[SO OVR WARN] so.{attr} is None / missing", flush=True)
            continue

    #this will only work if attach_storage_overrides already set S_obs on it
        print(f"[SO OVR] found so.{attr} key={getattr(r,'key',None)} "
              f"override={getattr(r,'use_storage_override',None)} "
              f"S_obs_none={getattr(r,'S_obs',None) is None} "
              f"last_t={getattr(r,'override_last_t',None)}", flush=True)

    #turn on debug prints from apply_storage_override
        try:
            r.debug_storage_override = 1
            r.debug_storage_until_t = override_last_so
            r.debug_storage_every = 1
        except Exception:
            pass

    ##debugg printssss, deleteeee
    for r in so_model.reservoir_list:
        print("[SO MODEL RES]", r.key,
          "use_storage_override=", getattr(r,"use_storage_override",None),
          "S_obs None?", getattr(r,"S_obs",None) is None,
          "override_last_t=", getattr(r,"override_last_t",None),
          "debug_storage_override=", getattr(r,"debug_storage_override",None))

    '''
    m = main_cy_obj.modelno
    trt=m.reservoir.

    m.trinity.debug_storage_override = 1
    m.trinity.debug_storage_until_t = 30
    m.trinity.debug_storage_every = 1
    print("TRT use_storage_override:", m.trinity.use_storage_override)
    print("TRT override_last_t:", m.trinity.override_last_t)
    print("TRT first 3 S_obs:", m.trinity.S_obs[:3])
    print("TRT first 3 S model:", m.trinity.S[:3])
    '''
    m = main_cy_obj.modelno

    #tnt=main_cy_obj.tntsys
    #debug printsss, deleteee
    for r in m.reservoir_list:
      print(r.key, "override?", getattr(r, "use_storage_override", None),
          "last_t", getattr(r, "override_last_t", None),
          "S_obs set?", r.S_obs is not None)
    
    for r in m.reservoir_list:
        if getattr(r, "use_storage_override", False):
            r.debug_storage_override = 1
            r.debug_storage_until_t = override_last_t
            r.debug_storage_every = 1
    
    print("len(m.reservoir_list) =", len(getattr(m, "reservoir_list", [])))
    if hasattr(m, "reservoir_list"):
      print("reservoir_list keys:", [getattr(r, "key", None) for r in m.reservoir_list][:50])
    
    #ALSO enable storage override debug for TRT in trt_list
    
    # Enable debug for TRT wherever it lives (trinity object)
    tri = getattr(main_cy_obj, "trinity", None)
    

#Try to locate TRT object in any likely list in tri
    trt_obj = None
    for nm in ["trt_list", "reservoir_list", "reservoirs"]:
        if tri is not None and hasattr(tri, nm):
            v = getattr(tri, nm)
            if isinstance(v, (list, tuple)):
                for r in v:
                    if str(getattr(r, "key", "")).strip().upper() == "TRT":
                        trt_obj = r
                        print(f"[TRT FOUND] in tri.{nm} | type={type(r)}")
                        break
            if trt_obj is not None:
                break

    if trt_obj is None:
        print("[TRT FOUND] NO TRT object found on tri lists")
    else:
        print("[TRT STATUS]",
              "use_storage_override=", getattr(trt_obj, "use_storage_override", None),
              "override_last_t=", getattr(trt_obj, "override_last_t", None),
              "S_obs set?", getattr(trt_obj, "S_obs", None) is not None,
              "S_obs[0:3]=", (getattr(trt_obj, "S_obs", None)[:3] if getattr(trt_obj, "S_obs", None) is not None else None))


    if tri is not None and hasattr(tri, "trt_list"):
        for r in (tri.trt_list or []):
            if str(getattr(r, "key", "")).strip().upper() == "TRT":
                r.debug_storage_override = 1
                r.debug_storage_until_t = override_last_t
                r.debug_storage_every = 1
                print("[DEBUG TRT] enabled debug on main_cy_obj.trinity.trt_list TRT")
                break
    else:
        print("[DEBUG TRT] main_cy_obj.trinity.trt_list not found")

    
    so  = main_cy_obj.modelso

# turn on debug + confirm attach for sn luis wherever they are, delete laterrr
    for nm in ["reservoir_list", "sls_list", "slf_list", "sanluis_list", "sl_list"]:
        if not hasattr(so, nm):
            continue
        lst = getattr(so, nm)
        if not isinstance(lst, (list, tuple)):
            continue

        for r in lst:
            k = str(getattr(r, "key", "")).strip().upper()

        #IF MODEL CALLS IT SL NOT SNLL
            if k in ["SLS", "SLF", "SL", "SNL"]:
                #attach exactly like attach_one() does
                #re-use the arrays already set via attach_storage_overrides if present
                if getattr(r, "S_obs", None) is None:
                    print(f"[OVR SO WARN] {k} found in so.{nm} but S_obs is None (not attached)")
                else:
                    r.use_storage_override = True
                    r.override_last_t = override_last_so

            #enable printsssss
                try:
                    r.debug_storage_override = 1
                    r.debug_storage_until_t = override_last_so
                    r.debug_storage_every = 1
                except Exception:
                    pass

                print(f"[OVR SO] {k} in so.{nm}: override={getattr(r,'use_storage_override',None)} "
                  f"S_obs? {getattr(r,'S_obs',None) is not None} last_t={getattr(r,'override_last_t',None)}")

    print("DEBUGGGG reservoir_list types (unique):", sorted({type(r).__name__ for r in m.reservoir_list}))
    if hasattr(m, "trt_list"):
        print("DEBUGGGG trt_list types (unique):", sorted({type(r).__name__ for r in m.trt_list}))

    sys.stdout.flush()

    print("[OVERRIDE] cutoff_date =", cutoff_date, "| override_last_t =", override_last_t,
      "| override_last_date =", m.df[0].index[override_last_t] if override_last_t >= 0 else None)

    # FIND SHASTAAAAA 
    shasta = None
    if hasattr(m, "reservoir_list"):
        for r in m.reservoir_list:
            k = str(getattr(r, "key", "")).strip().upper()
            n = str(getattr(r, "name", "")).strip().lower()
            if k == "SHA" or "shasta" in n:
                shasta = r
                break

    print("SHASTA found?", shasta is not None,
          "key=", getattr(shasta, "key", None),
          "name=", getattr(shasta, "name", None),
          "type=", type(shasta))
    sys.stdout.flush()

#TURN ON SHASTA STORAGE OVERRIDE DEBUG
    if shasta is not None:
        shasta.debug_storage_override = 1
        shasta.debug_storage_until_t = 30     # print through day 30
        shasta.debug_storage_every = 1        # print every day

        print("SHA debug set:",
          "debug_storage_override=", shasta.debug_storage_override,
          "until_t=", shasta.debug_storage_until_t,
          "every=", shasta.debug_storage_every)
        print("SHA override ON:",
          "use_storage_override=", getattr(shasta, "use_storage_override", None),
          "override_last_t=", getattr(shasta, "override_last_t", None))
        sys.stdout.flush()
    else:
        print("ERROR: Shasta not found in reservoir_list — printing first 50 keys to diagnose:")
        if hasattr(m, "reservoir_list"):
           print([getattr(r, "key", None) for r in m.reservoir_list][:50])
        sys.stdout.flush()


    print("\nDEBUGGGG modelno.T =", m.T)
    print("DEBUGGGG len(modelno.df[0]) =", len(m.df[0]))
    if hasattr(m, "df_short"):
      try:
        print("DEBUGGGG len(modelno.df_short[0]) =", len(m.df_short[0]))
        print("DEBUGGGG df_short first/last =", m.df_short[0].index[0], m.df_short[0].index[-1])
      except Exception as e:
        print("DEBUGGGG df_short error:", e)
    print("DEBUGGGG df first/last =", m.df[0].index[0], m.df[0].index[-1])

    print("Applied overrides to modelno through t =", override_last_t)
    print("Storage override active through t =", override_last_t, "date =", m.df[0].index[override_last_t])
    #END ADD 

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


