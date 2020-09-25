
eval "$(/home/santiagovargas/dev/anaconda3/etc/profile.d/conda.sh shell.bash hook)"
export SIGOPT_PROJECT=.sigopt_yaml
sigopt optimize train.py --dir DB3 --des morg --algo xgboost --sigopt --sigopt-file ./sigopt_experiments/xgboost.yml

