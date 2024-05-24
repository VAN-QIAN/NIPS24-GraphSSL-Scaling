MODEL=$1
DATASET=$2
CONFIG_FILE=$3

TASK_CMD="python3 ./run_model.py --task SSGCL --gpu_id 0 --model ${MODEL} --dataset ${DATASET} --gpu_id 0 --train_ratio ${CONFIG_FILE=} --downstream_ratio 0.1 --downstream_task loss --config_file random_config/config_1"

echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}