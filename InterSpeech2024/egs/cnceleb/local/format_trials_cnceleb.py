#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnceleb_root', help='cnceleb dir', type=str, default="CN-Celeb")
    parser.add_argument('--dst_trl_path', help='output trial path', type=str, default="new.trials")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    enroll_lst_path = os.path.join(args.cnceleb_root, "eval/lists/enroll.lst")
    raw_trl_path = os.path.join(args.cnceleb_root, "eval/lists/trials.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in enroll_lst:
        spk2wav_mapping[item[0]] = item[1]
    trials = np.loadtxt(raw_trl_path, str)

    with open(args.dst_trl_path, 'w') as f:
        for item in trials:
            enroll_path = spk2wav_mapping[item[0]]
            test_path = item[1]
            if args.apply_vad:
                enroll_path = enroll_path.replace('.wav', '.vad')
                test_path = test_path.replace('.wav', '.vad')
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))

