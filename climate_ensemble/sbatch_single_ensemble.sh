#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=800
#SBATCH --ntasks=1
#SBATCH --job-name=namehere
#SBATCH --output=job_status/out_%j.out
#SBATCH --error=job_status/err_%j.err
#SBATCH --array=1996-2024

module load python/3.8.8
source ../myenv/bin/activate

label=$1
echo "This is the first label variable:" "$label"
year=$SLURM_ARRAY_TASK_ID
echo "This is the first year variable:" "$year"
year_label="${year}_${label}"
echo "This is when you combine the two variables:" "$year_label"
echo "This is when you combine the two variables with brackets:" "${year_label}"

results_base='/proj/characklab/projects/danli/CALFEWS_results/'
results="${year_label}"

echo "${year_label}"
echo "${results}"
#sleep $(( (year - 1996) * 5 ))

# Create results directory
mkdir -p ${results_base}${results}

sed "s/sourcehere/${year_label}/" climate_ensemble/runtime_params_climate_tmp.ini > runtime_params_${year_label}.ini

echo "This is after sed command:" "${year_label}"



# Copy the modified runtime parameters file to results directory
cp runtime_params_${year_label}.ini ${results_base}${results}
mv ${results_base}${results}/runtime_params_${year_label}.ini ${results_base}${results}/runtime_params.ini

# Run the Python script
time python3 -W ignore run_main_cy.py $results 1 1 "${year}-9-30" ${results_base}${results}