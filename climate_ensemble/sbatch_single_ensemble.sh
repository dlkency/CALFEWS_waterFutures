#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=800
#SBATCH --ntasks=1
#SBATCH --job-name=namehere
#SBATCH --output=job_status/out_%j.out
#SBATCH --error=job_status/err_%j.err
#SBATCH --array=1996-2024

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

python -c "
import sys
assert sys.version_info[:2] == (3, 8), f'Python 3.8 is required, but found {sys.version}'
print('Confirmed: Python 3.8')
"

label=$1
year=$SLURM_ARRAY_TASK_ID
year_label="${year}_${label}"

results_base='/proj/characklab/projects/danli/CALFEWS_results/'
job_results_dir="${results_base}${year_label}"

echo "${year_label}"
echo "${job_results_dir}"

mkdir -p "${job_results_dir}"

sed "s/sourcehere/${year_label}/" climate_ensemble/runtime_params_climate_tmp.ini > "${job_results_dir}/runtime_params.ini"

time python3 -W ignore run_main_cy.py \
    "${job_results_dir}" \
    1 \
    1 \
    "${year}-09-30" \
    "${job_results_dir}" \
    "${year}-09-30"