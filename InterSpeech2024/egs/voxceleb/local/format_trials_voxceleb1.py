#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="voxceleb1")
    parser.add_argument('--src_trl_path', help='src_trials_path', type=str, default="voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trl_path', help='dst_trials_path', type=str, default="new_trials.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trl_path, dtype=str)

    f = open(args.dst_trl_path, 'w')
    for item in trials:
        # enroll_path = os.path.join(args.voxceleb1_root, item[1])
        # test_path = os.path.join(args.voxceleb1_root, item[2])
        enroll_path = item[1]
        test_path = item[2]
        if args.apply_vad:
            enroll_path = enroll_path.replace('.wav', '.vad')
            test_path = test_path.replace('.wav', '.vad')
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

