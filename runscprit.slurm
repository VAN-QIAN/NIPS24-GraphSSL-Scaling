#!/bin/bash
#SBATCH --job-name=run_split_enroot       # Job name
#SBATCH --partition=el8               # Partition
#SBATCH --gres=gpu:1                      # Request GPU resource
#SBATCH --time=00:10:00                   # Time limit hrs:min:sec
#SBATCH --output=slurm_logs/run_cci_%j.log  # Standard output and error log
#SBATCH --error=slurm_logs/run_cci_%j.err   # Error log

# Define the path to the script directory
SCRIPT_DIR="$HOME/scratch/KDD24-BGPM"
enroot start --root --rw --mount $SCRIPT_DIR:/data-valuation bgpm sh -c '
export http_proxy=http://proxy:8888 &&
export https_proxy=$http_proxy &&
rm -f /usr/lib64/libstdc++.so.6 &&
ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
ls -l /data-valuation &&
cd /data-valuation &&
pip list | grep torch &&
nvidia-smi &&
/bin/bash split_cci.sh'
