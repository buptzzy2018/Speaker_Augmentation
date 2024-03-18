import glob
import re
import fire
import torch


def main(src_path, last_n, dest_model):
    path_list = glob.glob('{}/epoch=*.ckpt'.format(src_path))
    path_list = sorted(path_list, key=lambda p: int(re.findall(r"(?<=epoch=)\d*(?=.ckpt)", p)[0]))
    path_list = path_list[-last_n:]
    print(path_list)
    avg = None
    assert last_n == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        if avg is None:
            avg = {}
            avg['state_dict'] = states
        else:
            for key in avg['state_dict'].keys():
                avg['state_dict'][key] += states[key]

    for key in avg['state_dict'].keys():
        if avg['state_dict'][key] is not None:
            avg['state_dict'][key] = torch.true_divide(avg['state_dict'][key], last_n)
    print('Saving to {}'.format(dest_model))
    torch.save(avg, dest_model)


if __name__ == '__main__':
    fire.Fire(main)
