import os
import argparse
import kaldiio
import numpy as np
from tqdm import tqdm


def compute_vector_mean(spk2utt, xvector_scp, spk_xvector_ark):
    # read spk2utt
    spk2utt_dict = {}
    with open(spk2utt, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split(' ')
            spk2utt_dict[line[0]] = line[1:]

    utt2embs = {}
    for utt, emb in kaldiio.load_scp_sequential(xvector_scp):
        utt2embs[utt] = emb

    spk_xvector_path = os.path.dirname(spk_xvector_ark)
    if not os.path.exists(spk_xvector_path):
        os.makedirs(spk_xvector_path)
    spk_xvector_ark = os.path.abspath(spk_xvector_ark)
    spk_xvector_scp = spk_xvector_ark[:-3] + "scp"
    with kaldiio.WriteHelper('ark,scp:' + spk_xvector_ark + "," + spk_xvector_scp) as writer:
        for spk in tqdm(spk2utt_dict.keys()):
            utts = spk2utt_dict[spk]
            mean_vec = np.sum([utt2embs[utt] for utt in utts], axis=0) / len(utts)
            writer(spk, mean_vec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute the mean of vector')
    parser.add_argument('--spk2utt', type=str, default='', help='spk2utt file')
    parser.add_argument('--xvector_scp', type=str, default='', help='xvector file (kaldi format)')
    parser.add_argument('--spk_xvector_ark', type=str, default='')
    args = parser.parse_args()
    compute_vector_mean(args.spk2utt, args.xvector_scp, args.spk_xvector_ark)
