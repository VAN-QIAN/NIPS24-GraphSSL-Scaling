models=("GraphMAE")
datasets=("ogbg-ppa")
ratio=("0.2" "0.3" "0.4" "0.5" )
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for i in "${ratio[@]}"; do
            config_file="random_config_new/config_${i}"
            job_name="${model}_${dataset}_${i}"
            log_dir="logs/${model}/${dataset}"
            mkdir -p "${log_dir}"
            sbatch --job-name="${job_name}" \
                   --output="${log_dir}/${job_name}_%j.log" \
                   --error="${log_dir}/${job_name}_%j.err" \
                   slurm_script_mvgrlg.sh "${model}" "${dataset}" "${i}"
            sleep 20
        done
    done
done