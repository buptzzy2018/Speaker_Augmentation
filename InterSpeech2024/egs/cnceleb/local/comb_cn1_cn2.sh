#!/bin/bash

# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cnceleb1_audio_dir=/data_root/CN-Celeb/data/
# cnceleb2_audio_dir=/data_root/CN-Celeb2/data/
min_duration=5
get_dur_nj=60
statistics_dir=statistics
store_data_dir=new_data

. local/parse_options.sh
set -e

mkdir -p $statistics_dir


# # combine the short audios for Cnceleb2
# cnceleb2_audio_dir=`realpath $cnceleb2_audio_dir`
# # get the paths of all the audio files
# find $cnceleb2_audio_dir -name "*.wav" | sort > ${statistics_dir}/cnceleb2_audio_path_list
# echo "combine audios for cnceleb2"
# bash local/combine_utt.sh --stage 0 \
#                     --ori_audio_dir ${cnceleb2_audio_dir} \
#                     --new_audio_dir ${store_data_dir} \
#                     --data_statistics_dir ${statistics_dir}/cnceleb2 \
#                     --audio_path_list ${statistics_dir}/cnceleb2_audio_path_list \
#                     --min_duration ${min_duration} \
#                     --get_dur_nj ${get_dur_nj}


# combine the short audios for Cnceleb1
cnceleb1_audio_dir=`realpath $cnceleb1_audio_dir`
# get the paths of all the audio files
find $cnceleb1_audio_dir -name "*.wav" | awk -F/ '{if($(NF-1)<"id00800"){print $0}}' | sort > ${statistics_dir}/cnceleb1_audio_path_list
echo "combine audios for cnceleb1_dev"
bash local/combine_utt.sh --stage 0 \
                    --ori_audio_dir ${cnceleb1_audio_dir} \
                    --new_audio_dir ${store_data_dir} \
                    --data_statistics_dir ${statistics_dir}/cnceleb1 \
                    --audio_path_list ${statistics_dir}/cnceleb1_audio_path_list \
                    --min_duration ${min_duration} \
                    --get_dur_nj ${get_dur_nj}

