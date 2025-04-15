#!/bin/bash

main_program="main_response.py"
conda_env="newtestrag"
log_folder="logs_test_100"
model_path="/root/autodl-tmp/new_model/model_bsize32_fb_r0.001_c5_10b1_r10"
current_time=$(date +'%Y-%m-%d_%H-%M-%S')

mkdir -p $log_folder


export NLTK_DATA=/home/LAB/maoqr/nltk_data
error_log="$log_folder/${current_time}_${dataset}_100_test_error.log"
output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"

python $main_program --model="/root/autodl-tmp/20250322_b16/model_en_bsize16_re-5_c5_1b100_25r_1ep_50000_rdm_r22" > $output_log
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
# output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
# python $main_program --model="/root/autodl-tmp/new_model/bge-base-en-v1.5" > $output_log
# current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_bsize32_fb_r0.001_c5_r3" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_bsize32_fb_r0.001_c5_r4" > $output_log
#current_time=$(date +'%Y-%m-%d_%H-%M-%S')
#output_log="$log_folder/${current_time}_${dataset}_100_test_output.log"
#python $main_program --model="/root/autodl-tmp/new_model/model_bsize32_fb_r0.001_c5_r5" > $output_log
