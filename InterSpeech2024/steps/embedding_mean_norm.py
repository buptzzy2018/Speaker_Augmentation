import os
import fire
import kaldiio
import numpy as np


def main(embedding_dir):
    scp_path = os.path.join(embedding_dir, 'xvector.scp')
    print("Calculate mean statistics from {}.".format(scp_path))

    vec_num = 0
    mean_vec = None

    for _, vec in kaldiio.load_scp_sequential(scp_path):
        if mean_vec is None:
            mean_vec = np.zeros_like(vec)
        mean_vec += vec
        vec_num += 1

    mean_vec /= vec_num

    mean_vec_path = os.path.join(embedding_dir, 'mean_vec.npy')
    np.save(mean_vec_path, mean_vec)


if __name__ == "__main__":
    fire.Fire(main)
