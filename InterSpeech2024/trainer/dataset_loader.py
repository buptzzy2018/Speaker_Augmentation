#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import numpy as np
import pandas as pd
import random
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset
import nlpaug.augmenter.audio as naa
import nlpaug.model.audio as nma



def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    audio, sr = load_wav_file(filename)
    audio = random_chunk(audio, max_frames, evalmode, num_eval)
    return audio, sr


def load_wav_file(filename):
    sr, audio = wavfile.read(filename)
    return audio, sr


def random_chunk(audio, max_frames, evalmode=False, num_eval=10):
    max_audio = max_frames
    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and num_eval == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return feat


def speed_perturb(waveform, sample_rate, label, num_spks, speed_par):
    """ Apply speed perturb to the data.
    """
    speeds = speed_par
    speed_idx = random.randint(1, len(speeds))
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        torch.from_numpy(waveform[np.newaxis, :]), sample_rate,
        [['speed', str(speeds[(speed_idx-1)])], ['rate', str(sample_rate)]])
    waveform = wav.numpy()[0]
    label = label + num_spks * speed_idx

    return waveform, label


class AugmentWAV(object):
    def __init__(self, musan_data_list_path, rirs_data_list_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
        self.noiselist = {}

        df = pd.read_csv(musan_data_list_path)
        augment_files = df["utt_paths"].values
        augment_types = df["speaker_name"].values
        for idx, file in enumerate(augment_files):
            if not augment_types[idx] in self.noiselist:
                self.noiselist[augment_types[idx]] = []
            self.noiselist[augment_types[idx]].append(file)
        df = pd.read_csv(rirs_data_list_path)
        self.rirs_files = df["utt_paths"].values

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        audio = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True) + audio
        return audio.astype(np.int16).astype(float)

    def reverberate(self, audio):
        rirs_file = random.choice(self.rirs_files)
        fs, rirs = wavfile.read(rirs_file)
        rirs = np.expand_dims(rirs.astype(float), 0)
        rirs = rirs / np.sqrt(np.sum(rirs**2))
        if rirs.ndim == audio.ndim:
            audio = signal.convolve(audio, rirs, mode='full')[:,:self.max_audio]
        return audio.astype(np.int16).astype(float)


class vtlpAug(naa.VtlpAug):
    def __init__(self, sampling_rate=16000, zone=(0.0, 1.0), coverage=1.0, fhi=4800, factor=(0.9, 1.1), 
        name='Vtlp_Aug', verbose=0, stateless=True):
        super().__init__(
            sampling_rate=sampling_rate,zone=zone, coverage=coverage, factor=factor, name=name, 
            verbose=verbose, stateless=stateless)

        self.fhi = fhi
        self.model = nma.Vtlp()

    def VTLP(self, data, sampling_rate, speaker_id, num_spks,warp_par):

        if self.duration is None:
            start_pos, end_pos = self.get_augment_range_by_coverage(data)
        else:
            start_pos, end_pos = self.get_augment_range_by_duration(data)

        warp_arr = warp_par
        warp_idx = random.randint(1,len(warp_par))
        warp_factor = warp_arr[warp_idx-1]

        if not self.stateless:
            self.start_pos, self.end_pos, self.aug_factor = start_pos, end_pos, warp_factor

        speaker_id = speaker_id + num_spks * warp_idx
        waveform = self.model.manipulate(data, start_pos=start_pos, end_pos=end_pos, sampling_rate=sampling_rate, warp_factor=warp_factor)

        return np.array(waveform).squeeze(), speaker_id


class Train_Dataset(Dataset):
    def __init__(self, data_list_path, aug_prob, speed_perturb, VTLP_Aug, warp_par, speed_par,max_frames, sample_rate,
                 musan_list_path=None, rirs_list_path=None, eval_mode=False, data_key_level=0):
        # load data list
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        self.speaker_number = len(np.unique(self.data_label))
        print("Train Dataset load {} speakers".format(self.speaker_number))
        print("Train Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = max_frames * sample_rate // 100
        self.aug_prob = aug_prob
        self.speed_perturb = speed_perturb
        self.VTLP_Aug = VTLP_Aug
        self.eval_mode = eval_mode
        self.data_key_level = data_key_level
        self.warp_par = warp_par
        self.speed_par = speed_par
        print('self.warp_par', self.warp_par)
        print('self.speed_par', self.speed_par)
                

        if aug_prob > 0:
            self.augment_wav = AugmentWAV(musan_list_path, rirs_list_path, max_frames=self.max_frames)
        if VTLP_Aug:
            self.vtlpAug = vtlpAug()

        self.label_dict = {}
        for idx, speaker_label in enumerate(self.data_label):
            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []
            self.label_dict[speaker_label].append(idx)
    
    def __getitem__(self, index):
        audio, sr = load_wav_file(self.data_list[index])
        label = self.data_label[index]

        if self.speed_perturb and self.VTLP_Aug:
            whether = random.randint(0, 1)
            if whether == 1:
                extend_type = random.randint(1, 2)
                if extend_type == 1:
                    audio, label = speed_perturb(audio, sr, label, self.speaker_number, self.speed_par)
                else:
                    audio, label = self.vtlpAug.VTLP(audio.astype(np.float), sr, label, self.speaker_number, self.warp_par)
                    label = label + self.speaker_number * 2

        elif self.VTLP_Aug:
            whether = random.randint(0, len(self.warp_par))
            if whether != 0:
                audio, label = self.vtlpAug.VTLP(audio.astype(np.float), sr, label, self.speaker_number,self.warp_par)
        
        elif self.speed_perturb:
            whether = random.randint(0, len(self.speed_par))
            if whether != 0:
                audio, label = speed_perturb(audio, sr, label, self.speaker_number,self.speed_par)

        audio = random_chunk(audio, self.max_frames)

        if self.aug_prob > random.random():
            augtype = random.randint(1, 4)
            if augtype == 1:
                audio = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio = self.augment_wav.additive_noise('music', audio)
            elif augtype == 3:
                audio = self.augment_wav.additive_noise('speech', audio)
            elif augtype == 4:
                audio = self.augment_wav.additive_noise('noise', audio)

        if self.eval_mode:
            data_path_sp = self.data_list[index].split('/')
            data_key = data_path_sp[-1]
            for i in range(2, self.data_key_level + 1):
                data_key = data_path_sp[-i] + '/' + data_key
            return torch.FloatTensor(audio), data_key

        return torch.FloatTensor(audio), label

    def __len__(self):
        return len(self.data_list)


class Dev_Dataset(Dataset):
    def __init__(self, data_list_path, eval_frames, num_eval=0, **kwargs):
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Dev Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Dev Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio, sr = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Test_Dataset(Dataset):
    def __init__(self, data_list, eval_frames, num_eval=0, **kwargs):
        # load data list
        self.data_list = data_list
        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio, sr = loadWAV(self.data_list[index][1], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_list[index][0]

    def __len__(self):
        return len(self.data_list)


def round_down(num, divisor):
    return num - (num % divisor)


class Train_Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size

    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data = self.label_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(np.random.permutation(len(data))[:numSeg], self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid = np.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        return iter([flattened_list[i] for i in mixmap])

    def __len__(self):
        return len(self.data_source)


if __name__ == "__main__":
    data, sr = loadWAV("test.wav", 100, evalmode=True)
    print(data.shape)
    data, sr = loadWAV("test.wav", 100, evalmode=False)
    print(data.shape)

    def plt_wav(data, name):
        import matplotlib.pyplot as plt
        x = [ i for i in range(len(data[0])) ]
        plt.plot(x, data[0])
        plt.savefig(name)
        plt.close()

    plt_wav(data, "raw.png")
    
    aug_tool = AugmentWAV("data/musan_list.csv", "data/rirs_list.csv", 100)

    audio = aug_tool.reverberate(data)
    plt_wav(audio, "reverb.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "music.png")

    audio = aug_tool.additive_noise('speech', data)
    plt_wav(audio, "speech.png")

    audio = aug_tool.additive_noise('noise', data)
    plt_wav(audio, "noise.png")
