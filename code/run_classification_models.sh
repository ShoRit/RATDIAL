#!/bin/bash

#SBATCH --job-name=shire-1
#SBATCH --gres=gpu:1                 
#SBATCH --time=1-00:00                  # Time limit in D-HH:MM
#SBATCH --output=/home/rdutt/slurm_outs/%j.out

TASK=$1
INFO=$2
DATASET=$3
MODEL=$4
FEWSHOT=$5
SEED=$6

source activate /home/rdutt/anaconda3/envs/SICON

python3 /data/shire/projects/TLDIAL/code/run_cls_explanations.py --mode ID --info $INFO --task $TASK --model_name $MODEL --dataset $DATASET --batch_size 16 --fewshot $FEWSHOT --seed $SEED  --do_train 1 --do_test 1 --turns 5 &

# python3 /data/shire/projects/TLDIAL/code/run_cls_explanations.py --mode ID  --info all --task res --model_name t5-large --dataset P4G --batch_size 16 --fewshot -1 --seed 0 --do_train 1 --do_test 1 --turns 5

