#!/bin/bash

SUNINE_ROOT=../..
config=conf/config.yaml
exp_dir=exp
ckpt_path=$exp_dir/checkpoints/avg_model.ckpt
xvec_path=$exp_dir/embeddings
eval_scp_path=wav.scp
cuda_device="0"

. ./local/parse_options.sh || exit 1;
set -e

cuda_array=($(echo $cuda_device | tr "," "\n"))
nj=${#cuda_array[@]}

log_dir=$xvec_path/split_log
mkdir -p $log_dir

data_num=$(wc -l ${eval_scp_path} | awk '{print $1}')
subfile_num=$(($data_num / $nj + 1))
split -l ${subfile_num} -d -a 3 ${eval_scp_path} ${log_dir}/split_

for job in $(seq 0 $(($nj - 1))); do
  suffix=$(printf '%03d' $job)
  data_list_subfile=${log_dir}/split_${suffix}
  log_split=${log_dir}/extract_split_${suffix}.log

  split_xvec_path=$xvec_path/xvec_split_${suffix}
  mkdir -p $split_xvec_path

  CUDA_VISIBLE_DEVICES=${cuda_array[$job]} python3 -W ignore $SUNINE_ROOT/main.py \
      --config $config \
      --checkpoint_path $ckpt_path \
      --evaluate \
      --extract \
      --exp_dir $exp_dir \
      --xvec_path $split_xvec_path \
      --eval_scp_path $data_list_subfile \
      --gpus 1 \
      --num_workers 2 \
      > $log_split 2>&1 &
done
wait
cat $xvec_path/xvec_split_*/xvector.scp > $xvec_path/xvector.scp
