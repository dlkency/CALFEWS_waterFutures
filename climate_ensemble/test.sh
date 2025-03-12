#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=800
#SBATCH --ntasks=1
#SBATCH --job-name=namehere
#SBATCH --output=job_status/out_%j.out
#SBATCH --error=job_status/err_%j.err


module load python/3.8.8
source ../myenv/bin/activate

label=$1
year = 1996
results="${label}_${year}/"

sed 's/sourcehere/'$label'/' climate_ensemble/runtime_params_climate_tmp.ini > runtime_params.ini
results_base='/proj/characklab/projects/danli/CALFEWS_results/'

mkdir -p ${results_base}${results}
cp runtime_params.ini ${results_base}${results}

time python3 -W ignore run_main_cy.py $results 1 1 "${year}-9-30"