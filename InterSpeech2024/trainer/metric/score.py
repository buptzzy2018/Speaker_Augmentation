#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from tqdm import tqdm
from numba import jit
try:
    from .compute_eer import compute_eer
    from .tuneThreshold import *
except:
    from compute_eer import compute_eer
    from tuneThreshold import *


@jit
def cosine_score(trials, index_mapping, eval_vectors, scores='', apply_metric=True):
    all_scores = []
    all_labels = []

    f = open(scores, 'w') if os.path.exists(os.path.dirname(scores)) else open(os.devnull, 'w')
    for item in tqdm(trials):
        all_labels.append(int(item[0]))
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        dim = len(enroll_vector)
        score = enroll_vector.dot(test_vector.T)
        norm = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = dim * score / norm
        all_scores.append(score)
        f.write(item[0] + ' ' + item[1] + ' ' + item[2] + ' ' + str(score) + '\n')
    f.close()

    if apply_metric:
        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)
        return eer, th, mindcf_easy, mindcf_hard


@jit
def PLDA_score(trials, index_mapping, eval_vectors, plda_analyzer, scores='', apply_metric=True):
    all_scores = []
    all_labels = []

    f = open(scores, 'w') if os.path.exists(os.path.dirname(scores)) else open(os.devnull, 'w')
    for item in tqdm(trials):
        all_labels.append(int(item[0]))
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = plda_analyzer.NLScore(enroll_vector, test_vector)
        all_scores.append(score)
        f.write(item[0] + ' ' + item[1] + ' ' + item[2] + ' ' + str(score) + '\n')
    f.close()

    if apply_metric:
        eer, th = compute_eer(all_scores, all_labels)

        c_miss = 1
        c_fa = 1
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
        mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)
        return eer, th, mindcf_easy, mindcf_hard
