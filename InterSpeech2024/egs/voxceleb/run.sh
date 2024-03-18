#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.

SUNINE_ROOT=../..
voxceleb1_root=/ssd.m2/VoxCeleb/voxceleb1/vox1_dev_wav 
voxceleb2_root=
musan_path=/ssd.m2/musan
rirs_path=/ssd.m2/RIRS_NOISES/simulated_rirs

config=conf/ResNet34_ASP_AAMSoftmax.yaml
exp_dir="exp/resnet_asp_aam"
ckpt_path=
cuda_device=0,1  # Change the `--gpus` parameter correspondingly

stage=4
stop_stage=4


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
  # flac to wav
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/data \
          --speaker_level 1

  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb2_path/data \
          --speaker_level 1
 
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/enroll \
          --speaker_level 0
  
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/test \
          --speaker_level 0
fi


# In our experiment, we found that VAD seems useless.
# Here directly skip this stage.
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
  # compute VAD for each dataset
  echo Compute VAD on cnceleb1
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 16

  echo Compute VAD on cnceleb2
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb2_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 16

  echo Compute VAD on enroll
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/enroll \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 16

  echo Compute VAD on test
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/test \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 16
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
  # prepare data

  echo "prepare training data ..."
  [ -d data ] && rm -r data
  mkdir -p data/train

  for spk in $voxceleb1_root/*; do
    find ${spk}/ -name "*.wav" | \
    awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >> data/train/wav.scp
  done

  awk '{print $1}' data/train/wav.scp | awk -F "/" '{print $0,$1}' > data/train/utt2spk
  local/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

  echo "prepare testing data ..."

  mkdir -p data/eval
  find ${voxceleb1_root}/../test -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort > data/eval/vox1_test.scp

  echo "prepare evaluation trials ..."
  mkdir -p data/trials
  python3 local/format_trials_voxceleb1.py \
          --voxceleb1_root $voxceleb1_root \
          --src_trl_path conf/List_of_trial_pairs-VoxCeleb1-Clean.txt \
          --dst_trl_path data/trials/VoxCeleb1-Clean.lst 

fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
  # prepare data for model training

  echo Build train list
  python3 $SUNINE_ROOT/steps/build_datalist.py \
          --data_dir $voxceleb1_root \
          --extension wav \
          --speaker_level 1 \
          --data_list_path data/train_lst.csv

  echo Build $musan_path list
  python3 $SUNINE_ROOT/steps/build_datalist.py \
          --data_dir $musan_path \
          --extension wav \
          --data_list_path data/musan_lst.csv

  echo Build $rirs_path list
  python3 $SUNINE_ROOT/steps/build_datalist.py \
          --data_dir $rirs_path \
          --extension wav \
          --data_list_path data/rirs_lst.csv
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ];then
  # model training
  echo "model training ..."

  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SUNINE_ROOT/main.py \
          --config $config \
          ${ckpt_path:+--checkpoint_path $ckpt_path} \
          --exp_dir $exp_dir \
          --train_list_path data/train_lst.csv \
          --musan_list_path data/musan_lst.csv \
          --rirs_list_path data/rirs_lst.csv \
          --eval_scp_path data/eval/vox1_test.scp \
          --trials_path data/trials/VoxCeleb1-Clean.lst \
          --distributed_backend dp \
          --reload_dataloaders_every_epoch \
          --gpus 2
fi


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ];then
  # average checkpoints

  echo "average checkpoints ..."
  avg_model=$exp_dir/checkpoints/avg_model.ckpt
  last_n=10  # Change the `last_n` parameter of config.yaml correspondingly

  python $SUNINE_ROOT/steps/average_checkpoints.py \
      --src_path $exp_dir/checkpoints \
      --dest_model $avg_model \
      --last_n $last_n
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ];then
  echo "extract embedding ..."

  ckpt_path=$exp_dir/checkpoints/avg_model.ckpt
  echo $ckpt_path

  for dset in eval; do
    xvec_path=$exp_dir/embeddings/$dset
    [ -d $xvec_path ] && rm -r $xvec_path
    mkdir -p $xvec_path

    local/extract.sh \
        --SUNINE_ROOT $SUNINE_ROOT \
        --config $config \
        --exp_dir $exp_dir \
        --ckpt_path $ckpt_path \
        --xvec_path $xvec_path \
        --eval_scp_path data/$dset/vox1_test.scp \
        --cuda_device $cuda_device
  done
fi


if [ $stage -le 7 ] && [ $stop_stage -ge 7 ];then
  # evaluation

  mkdir -p $exp_dir/scores/

  for trials in VoxCeleb1-Clean ; do
    echo Evaluate $trials
    python -W ignore $SUNINE_ROOT/trainer/metric/compute_score.py \
            --trials_path data/trials/$trials.lst \
            --eval_scp_path $exp_dir/embeddings/eval/xvector.scp \
            --scores_path $exp_dir/scores/$trials.foo
  done
fi


if [ $stage -le 8 ] && [ $stop_stage -ge 8 ];then
  # score normalization
  
  scores="VoxCeleb1-Clean" # VoxCeleb1-Clean VoxCeleb1-E-Clean VoxCeleb1-H-Clean
  score_norm_method="asnorm"  # asnorm/snorm
  cohort_set=train
  top_n=300

  # do mean normalization for evaluation embeddings
  echo "do mean normalization for evaluation embeddings"
  python $SUNINE_ROOT/steps/embedding_mean_norm.py $exp_dir/embeddings/train

  # compute mean vector of training speaker (cohort)
  echo "compute mean vector of training speaker (cohort)"
  python $SUNINE_ROOT/steps/vector_mean.py \
      --spk2utt data/$cohort_set/spk2utt \
      --xvector_scp $exp_dir/embeddings/${cohort_set}/xvector.scp \
      --spk_xvector_ark $exp_dir/embeddings/${cohort_set}/spk_xvector.ark
  
  echo "compute norm score"
  output_name=${scores}_${score_norm_method}_${top_n}
  python $SUNINE_ROOT/trainer/backend/score_norm.py \
      --score_norm_method $score_norm_method \
      --top_n $top_n \
      --trial_score_file $exp_dir/scores/${scores}.foo \
      --score_norm_file $exp_dir/scores/${output_name}.foo \
      --cohort_emb_scp $exp_dir/embeddings/${cohort_set}/spk_xvector.scp \
      --eval_emb_scp $exp_dir/embeddings/eval/xvector.scp \
      --mean_vec_path $exp_dir/embeddings/train/mean_vec.npy

  echo "compute metrics"
  for score in $exp_dir/scores/$output_name.foo; do
      echo $score
      python $SUNINE_ROOT/trainer/metric/compute_metrics.py $score
  done
fi
