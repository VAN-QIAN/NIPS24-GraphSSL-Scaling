#!/bin/bash

ratio=0.2

models=('GraphCL' 'JOAO') 
datasets=("ogbg-ppa")
tasks=('original' 'loss')
num_layers_list=(1 2 3 4 5)
hidden_dim_list=(32)
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --downstream_ratio 0.1 --downstream_task TASK_PLACEHOLDER --gpu_id 6 --config_file CONFIG_FILE_PLACEHOLDER --gpu_id 5'
commands=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do  
        for task in "${tasks[@]}"; do  
            for num_layers in "${num_layers_list[@]}"; do
                for hidden_dim in "${hidden_dim_list[@]}"; do
                    config_file="random_config/$model/${model}_${num_layers}_${hidden_dim}"
                    command="${template/MODEL_PLACEHOLDER/$model}"
                    command="${command/DATASET_PLACEHOLDER/$dataset}"
                    command="${command/RATIO_PLACEHOLDER/1}"
                    command="${command/TASK_PLACEHOLDER/$task}"
                    command="${command/CONFIG_FILE_PLACEHOLDER/$config_file}"
                    commands+=("$command")
                done
            done
        done
    done
done

for command in "${commands[@]}"; do
    echo $command
    eval $command
done
# parallel -j 1 eval ::: "${commands[@]}"