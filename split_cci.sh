#!/bin/bash

ratio=("1" "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

models=('GraphMAE')
datasets=("reddit_threads" "github_stargazers")
#datasets=("MUTAG" "MCF-7" "MOLT-4" "P388" "ZINC_full" "reddit_threads" "github_stargazers")
template="python3 /KDD24-BGPM/run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --train_ratio RATIO_PLACEHOLDER --downstream_ratio 0.1 --downstream_task loss --config_file random_config/config_1"
commands=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for i in $(seq $ratio $ratio 1); do
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command/RATIO_PLACEHOLDER/$i}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}"; do
    echo $command
    eval $command
done
