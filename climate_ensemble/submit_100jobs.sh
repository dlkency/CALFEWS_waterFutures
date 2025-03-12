#!/bin/bash

total_jobs=100

years=({1996..2024})  # Create an array of years from 1996 to 2024

for year in "${years[@]}"; do
  for (( label=1; label<=total_jobs; label++ )); do
    results="climate_ensemble/${label}_${year}/"

    # Generate a unique script for each job
    sed 's/namehere/'$label'/' climate_ensemble/sbatch_single_longleaf.sh > climate_ensemble/manyrun/sbatch_single_longleaf_${label}_${year}.sh
    sed -i 's/outhere/out_'${label}_${year}'.txt/' climate_ensemble/manyrun/sbatch_single_longleaf_${label}_${year}.sh
    sed -i 's/errhere/err_'${label}_${year}'.txt/' climate_ensemble/manyrun/sbatch_single_longleaf_${label}_${year}.sh

    # Submit the job
    jobid=$(sbatch --parsable climate_ensemble/manyrun/sbatch_single_longleaf_${label}_${year}.sh $label $results $year)

  done
done