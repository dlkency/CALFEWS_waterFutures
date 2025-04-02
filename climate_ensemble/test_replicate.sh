#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH --mem=800
#SBATCH --ntasks=1
#SBATCH --job-name=namehere

module load python/3.8.8
source ../myenv/bin/activate

time python -W ignore run_main_cy.py results/1_2013 1 1 "2013-9-30"