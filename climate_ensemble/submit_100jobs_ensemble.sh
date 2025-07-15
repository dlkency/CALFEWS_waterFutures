#!/bin/bash

total_jobs=100 

for (( label=1; label<=total_jobs; label++ )); do
    echo "Submitting array 1996-2024 for label: $label"
    # Submit the entire array for each label
    sbatch --array=1996-2024 climate_ensemble/sbatch_single_ensemble.sh $label 
done