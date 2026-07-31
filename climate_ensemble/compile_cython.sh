#!/bin/bash
#SBATCH --job-name=compile_cython
#SBATCH --output=job_status/compile_cython_%j.out
#SBATCH --error=job_status/compile_cython_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv


# Show the environment used for compilation
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_PREFIX"
echo "Python executable: $(which python)"
python --version

# Confirm that the environment uses Python 3.8
python -c "
import sys
assert sys.version_info[:2] == (3, 8), \
    f'Python 3.8 is required, but found {sys.version}'
print('Confirmed: Python 3.8')
"

# Remove previous compilation outputs
rm -rf build
find . -name "*.so" -delete

# Compile the Cython extensions
python setup_cy.py build_ext --inplace

echo "Cython compilation completed successfully."