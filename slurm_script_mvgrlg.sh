#!/bin/bash
#SBATCH --partition=el8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x/%x_%j.log
#SBATCH --error=%x/%x_%j.err

enroot start --root --rw --mount $HOME/scratch/KDD24-BGPM:/data-valuation bgpm sh -c '
ls -l &&
rm -f /usr/lib64/libstdc++.so.6 &&
ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
ls -l /data-valuation &&
cd /data-valuation &&

pip list | grep torch &&
nvidia-smi &&
/bin/bash ./run_task.sh "'"$1"'" "'"$2"'" "'"$3"'" '