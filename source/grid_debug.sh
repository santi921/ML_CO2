#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C knl
#SBATCH --time 00:30:00
#SBATCH -o ./term_out/out.%j

#SBATCH --mail-user=santiagovargas921@gmail.com

conda activate base
python train.py --des aval --grid --dir DB3 --algo grad
