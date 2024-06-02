#!/bin/bash
#SBATCH --partition=el8       # Partition name
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --time=06:00:00            # Time limit
#SBATCH --output=%x/%x_%j.log      # Standard output and error log
#SBATCH --error=%x/%x_%j.err       # Standard error

# Load any modules or source your software profile if necessary

# Starting the environment with necessary bindings and execution script
SCRIPT_DIR="$HOME/barn/KDD24-BGPM"
LOG_DIR="$HOME/scratch/log"
RAW_DATA_DIR="$HOME/scratch/raw_data"

# Start the enroot container and run the split.sh script
enroot start --root --rw \
    --mount $SCRIPT_DIR:/KDD24-BGPM\
    --mount $LOG_DIR:/scratch/log\
    --mount $RAW_DATA_DIR:/scratch/raw_data\
    bgpm_new sh -c '
ls -l &&
rm -f /usr/lib64/libstdc++.so.6 &&
ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
cd /KDD24-BGPM &&
pip list | grep torch &&
/bin/bash /KDD24-BGPM/script.sh "'"$1"'" "'"$2"'" "'"$3"'"
'

#
# rm -f /usr/lib64/libstdc++.so.6 &&
# ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
