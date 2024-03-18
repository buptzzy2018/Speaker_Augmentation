import fire
import kaldiio
import numpy as np

from score import cosine_score


def main(trials_path, eval_scp_path, scores_path):
    trials = np.loadtxt(trials_path, dtype=str)
    index_mapping = {}
    eval_vectors = []
    for utt, embedding in kaldiio.load_scp_sequential(eval_scp_path):
        index_mapping[utt] = len(eval_vectors)
        eval_vectors.append(embedding)
    eval_vectors = np.array(eval_vectors)
    eer, th, mindcf_e, mindcf_h = cosine_score(trials, index_mapping, eval_vectors, scores=scores_path)
    print('Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}'.format(eer * 100, mindcf_e, mindcf_h))


if __name__ == '__main__':
    fire.Fire(main)
