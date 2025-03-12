#!/bin/bash

total_jobs=100  
years=({1996..2024})

for (( label=1; label<=total_jobs; label++ )); do
    # Submit the entire array for each label
    sbatch --array=1996-2024 climate_ensemble/sbatch_single_ensemble.sh $results
done