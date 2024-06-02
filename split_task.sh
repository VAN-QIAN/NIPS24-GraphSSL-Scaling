#!/bin/bash

# 定义模型和数据集数组
# models=("MVGRLG" "GraphMAE" )
models=("GraphCL" "JOAO") # "GraphCL" "MVGRLG" "InfoGraph") # 
datasets=("ogbg-ppa")
num_layers_list=(1 2 3 4 5)
hidden_dim_list=(32 64 128 256)
r=0.1
# 循环遍历每个模型和数据集，并为每个配置生成并提交一个Slurm作业
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for num_layers in "${num_layers_list[@]}"; do
            for hidden_dim in "${hidden_dim_list[@]}"; do
                config_file="random_config/${model}/${model}_${num_layers}_${hidden_dim}"
                job_name="${model}_${dataset}_${num_layers}_${hidden_dim}_original"
                log_dir="logs/${model}/${dataset}/model"
                mkdir -p "${log_dir}"
                sbatch --job-name="${job_name}" \
                    --output="${log_dir}/${job_name}_%j.log" \
                    --error="${log_dir}/${job_name}_%j.err" \
                    slurm.sh "${model}" "${dataset}" "${config_file}"
                sleep 10
            done
        done
    done
done