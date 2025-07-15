#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=800
#SBATCH --ntasks=1
#SBATCH --job-name=namehere
#SBATCH --output=job_status/out_%j.out
#SBATCH --error=job_status/err_%j.err


module load python/3.8.8
source ../myenv/bin/activate


# Run the Python script
time python3 -W ignore make_financial_data.py 