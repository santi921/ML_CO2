#!/bin/bash
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C knl
#SBATCH --time 48:00:00
#SBATCH -o ./term_out/out.%j
#SBATCH --mail-user=santiagovargas921@gmail.com

conda activate base

python train.py --des morg --grid --dir DB3 --algo grad
python train.py --des morg --grid --dir DB3 --algo rf


python train.py --des rdkit --grid --dir DB3 --algo grad
python train.py --des rdkit --grid --dir DB3 --algo rf

python train.py --des layer --grid --dir DB3 --algo grad
python train.py --des layer --grid --dir DB3 --algo rf

python train.py --des aval --grid --dir DB3 --algo grad
python train.py --des aval --grid --dir DB3 --algo rf

python train.py --des vae --grid --dir DB3 --algo grad
python train.py --des vae --grid --dir DB3 --algo rf

python train.py --des auto --grid --dir DB3 --algo grad
python train.py --des auto --grid --dir DB3 --algo rf

