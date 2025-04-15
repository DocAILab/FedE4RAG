#!/bin/sh
# Specify number of CPUs and GPUs
#SBATCH -c 4
#SBATCH --gres=gpu:V100:4

main_program="main.py"
conda_env="/home/fedrag/env"
log_folder="logs"
dataset="law"
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
. activate /home/fedrag/env  # For POSIX shell, we use `.` to activate conda environment
mkdir -p $log_folder

error_log="$log_folder/${current_time}_${dataset}_error.log"
output_log="$log_folder/${current_time}_${dataset}_output.log"
python $main_program > $output_log 2> $error_log