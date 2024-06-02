#!/bin/bash

# Assign variables
MODEL=$1
DATASET=$2
CONFIG_FILE=$3

# Define the task command
TASK_CMD="python3 ./run_model.py --task SSGCL --model ${MODEL} --dataset ${DATASET} --gpu_id 0 --config_file ${CONFIG_FILE} --downstream_ratio 0.1 --downstream_task original"

# Execute the task
echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}