import os
import fire
from tqdm import tqdm

from compute_eer import compute_eer
from tuneThreshold import *


def main(scores_file):
    all_scores = []
    all_labels = []

    with open(scores_file) as f:
        for line in tqdm(f.readlines()):
            tokens = line.strip().split()
            assert len(tokens) == 4
            all_scores.append(float(tokens[3]))
            all_labels.append(int(tokens[0]))

    eer, th = compute_eer(all_scores, all_labels)

    c_miss = 1
    c_fa = 1
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    mindcf_easy, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, c_miss, c_fa)
    mindcf_hard, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.001, c_miss, c_fa)

    print("\n----- {} -----".format(os.path.basename(scores_file)))
    print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer * 100, mindcf_easy, mindcf_hard))


if __name__ == "__main__":
    fire.Fire(main)
