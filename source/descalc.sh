
#!/bin/bash
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C knl
#SBATCH --time 30:00:00
#SBATCH -o ./term_out/out.%j
#SBATCH --mail-user=santiagovargas921@gmail.com

conda activate base
python ./descalc.py --dir DB3 --desc rdkit 
python ./descalc.py --dir DB3 --desc layer
python ./descalc.py --dir DB3 --desc aval 
python ./descalc.py --dir DB3 --desc morg 
python ./descalc.py --dir DB3 --desc persist
 



