#!/bin/bash

ratio=("1" "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

models=('GraphMAE')
datasets=( "reddit_threads")
template='singularity exec --nv /data/qianMa/SIF/bgpmv116.sif  python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --train_ratio RATIO_PLACEHOLDER --config_file random_config/mvgrlg'
commands=()

for i in ${ratio[@]}; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command/RATIO_PLACEHOLDER/$i}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}";do
    echo $command
done
parallel -j 1 eval ::: "${commands[@]}"
