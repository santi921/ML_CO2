eval "$(conda shell.bash hook)"
conda activate tf_gpu
conda list

# set this for the experiment to be used
#export SIGOPT_PROJECT=.sigopt_yaml

# steps: set python command with right descriptor, energy band, yaml, db
# specify the right yaml file
# export the right experiment file



export SIGOPT_PROJECT=ml-co2-db3-diff-morg-sgd
#export SIGOPT_PROJECT=ml_co2
sigopt optimize train.py --dir DB3 --des morg --algo sgd --sigopt --diff --sigopt-file ./sigopt_experiments/sgd.yml
#sigopt run train.py --dir DB3 --des morg --algo sgd --sigopt
#export SIGOPT_PROJECT=ml-co2-db3-diff-rf
#sigopt optimize train.py --dir DB3 --des morg --algo rf --sigopt --diff --sigopt-file ./sigopt_experiments/rf.yml

#export SIGOPT_PROJECT=ml-co2-rdkit-db3-diff-grad
#sigopt optimize train.py --dir DB3 --des rdkit --algo grad --sigopt --diff --sigopt-file ./sigopt_experiments/grad.yml


# set --sigopt-file, the choices are:
# ./sigopt_experiments/xgboost.yml
#./sigopt_experiments/grad.yml
#./sigopt_experiments/svr.yml
#./sigopt_experiments/bayes.yml
#./sigopt_experiments/gaussian.yml
#./sigopt_experiments/rf.yml
#./sigopt_experiments/sgd.yml


#sigopt optimize train.py --dir DB3 --des morg --algo xgboost --sigopt --diff --sigopt-file ./sigopt_experiments/xgboost.yml

