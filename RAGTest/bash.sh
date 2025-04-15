#!/bin/bash

main_program="main_100_test.py"
conda_env="newtestrag"
log_folder="logs_test_100"
model_path="/root/autodl-tmp/new_model/model_bsize32_fb_r0.001_c5_10b1_r10"
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

mkdir -p $log_folder


export NLTK_DATA=/home/LAB/maoqr/nltk_data
error_log="$log_folder/${current_time}_${dataset}_100_test_error.log"
output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"

python $main_program --model="/root/autodl-tmp/new_model/bge-base-en-v1.5" > $output_log
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
python $main_program --model="/root/autodl-tmp/new_model/bge-large-en-v1.5" > $output_log
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/20250303/model_en_bsize32_re-5_c5_1b1_10r_1ep_50000_rdm_r8" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/20250303/model_en_bsize32_re-5_c5_1b1_10r_1ep_50000_rdm_r9" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_en_bsize32_re-2_c5_1b1_10r_5ep_r6" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_en_bsize32_re-2_c5_1b1_10r_5ep_r7" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_en_bsize32_re-2_c5_1b1_10r_5ep_r8" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_en_bsize32_re-2_c5_1b1_10r_5ep_r9" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_en_bsize32_re-2_c5_1b1_10r_5ep_r10" > $output_log
#
parent_folder="/root/autodl-tmp/20250318"
for dir in "$parent_folder"/*/; do
    if [ -d "$dir" ]; then
        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
        model_path=$(realpath "$dir")
        python "$main_program" --model="$model_path" > "$output_log"
    fi
done

parent_folder="/root/autodl-tmp/20250322"
for dir in "$parent_folder"/*/; do
    if [ -d "$dir" ]; then
        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
        model_path=$(realpath "$dir")
        python "$main_program" --model="$model_path" > "$output_log"
    fi
done
#parent_folder="/root/autodl-tmp/20250322"
#for dir in "$parent_folder"/*/; do
#    if [ -d "$dir" ]; then
#        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#        model_path=$(realpath "$dir")
#        python "$main_program" --model="$model_path" > "$output_log"
#    fi
#done

#main_program="test_upload.py"
#parent_folder="/root/autodl-tmp/20250322_b16"
#for dir in "$parent_folder"/*/; do
#    if [ -d "$dir" ]; then
#        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#        model_path=$(realpath "$dir")
#        python "$main_program" --model="$model_path" > "$output_log"
#    fi
#done
##
#parent_folder="/root/autodl-tmp/20250215"
#for dir in "$parent_folder"/*/; do
#    if [ -d "$dir" ]; then
#        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#        model_path=$(realpath "$dir")
#        python "$main_program" --model="$model_path" > "$output_log"
#    fi
#done
#
#parent_folder="/root/autodl-tmp/20250221"
#for dir in "$parent_folder"/*/; do
#    if [ -d "$dir" ]; then
#        current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#        output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#        model_path=$(realpath "$dir")
#        python "$main_program" --model="$model_path" > "$output_log"
#    fi
#done