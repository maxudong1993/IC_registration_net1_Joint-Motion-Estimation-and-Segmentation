#!/bin/bash
#SBATCH --nodes=4
#SBATCH --job-name=registration_test
#SBATCH --partition gpu
module list
source activate python37 
python main.py
source deactivate
