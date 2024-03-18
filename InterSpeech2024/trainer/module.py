#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

import kaldiio
import importlib
from tqdm import tqdm
from collections import OrderedDict

from .utils import PreEmphasis
from .dataset_loader import Train_Dataset, Test_Dataset, Train_Sampler
from .schedulers import MarginScheduler
from .metric import cosine_score


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # load trials and data list
        if os.path.exists(self.hparams.get('trials_path', '')):
            self.trials = np.loadtxt(self.hparams.trials_path, dtype=str)
            enroll_num = len(np.unique(self.trials.T[1]))
            test_num = len(np.unique(self.trials.T[2]))
            print("number of enroll: ", enroll_num)
            print("number of test: ", test_num)
        if os.path.exists(self.hparams.get('eval_scp_path', '')):
            self.eval_data = {}
            with open(self.hparams.eval_scp_path) as f:
                for line in f:
                    item = line.strip().split()
                    self.eval_data[item[0]] = item[1]
        if os.path.exists(self.hparams.get('train_list_path', '')):
            df = pd.read_csv(self.hparams.train_list_path)
            speaker = np.unique(df["utt_spk_int_labels"].values)
            self.hparams.num_classes = len(speaker)

            print("Number of Training Speaker classes is: {}".format(self.hparams.num_classes))
            self.batch_num = len(df) // self.hparams.batch_size
            print('batch num (loader size):', self.batch_num)
            
            if self.hparams.speed_perturb and self.hparams.VTLP:
                self.hparams.num_classes *= 5 

            elif self.hparams.speed_perturb:
                self.hparams.num_classes *= self.hparams.num_sp 

            elif self.hparams.VTLP:
                self.hparams.num_classes *= self.hparams.num_vtlp

        print('speed_perturb:', self.hparams.speed_perturb)
        print('VTLP:', self.hparams.VTLP)
        
        # Network information Report
        print("Network Type: ", self.hparams.nnet_type)
        print("Pooling Type: ", self.hparams.pooling_type)
        print("Embedding Dim: ", self.hparams.embedding_dim)

        #########################
        ### Network Structure ###
        #########################

        # 1. Acoustic Feature
        sr = self.hparams.sample_rate
        print('sample rate: ', sr)
        self.mel_trans = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512, 
                                                     win_length=sr * 25 // 1000, hop_length=sr * 10 // 1000, 
                                                     window_fn=torch.hamming_window, n_mels=self.hparams.n_mels)
                )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # 2. Speaker_Encoder
        Speaker_Encoder = importlib.import_module('trainer.nnet.' + self.hparams.nnet_type).__getattribute__('Speaker_Encoder')
        self.speaker_encoder = Speaker_Encoder(**dict(self.hparams))

        # 3. Loss / Classifier
        if not self.hparams.get('evaluate', False):
            LossFunction = importlib.import_module('trainer.loss.' + self.hparams.loss_type).__getattribute__('LossFunction')
            self.loss = LossFunction(**dict(self.hparams))

            if self.hparams.margin_scheduler['update_margin']:
                initial_margin = self.hparams.margin_scheduler['initial_margin']
                final_margin = self.hparams.margin_scheduler['final_margin']
                increase_start_epoch = self.hparams.margin_scheduler['increase_start_epoch']
                fix_start_epoch = self.hparams.margin_scheduler['fix_start_epoch']
                increase_type = self.hparams.margin_scheduler['increase_type']
                print("Margin Scheduler:")
                print("\tinitial margin:", initial_margin)
                print("\tfinal margin:", final_margin)
                print("\tincrease start epoch:", increase_start_epoch)
                print("\tfix start epoch:", fix_start_epoch)
                print("\tincrease type:", increase_type)
                self.margin_scheduler = MarginScheduler(self.loss, self.batch_num, increase_start_epoch, 
                                             fix_start_epoch, initial_margin, final_margin, True, increase_type)

    def forward(self, x, label):
        x = self.extract_speaker_embedding(x)
        x = x.reshape(-1, self.hparams.nPerSpeaker, self.hparams.embedding_dim)
        loss, acc = self.loss(x, label)
        return loss.mean(), acc

    def extract_speaker_embedding(self, data):
        x = data.reshape(-1, data.size()[-1])
        x = self.mel_trans(x) + 1e-6
        x = x.log()
        x = self.instancenorm(x)
        x = self.speaker_encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        if self.hparams.margin_scheduler['update_margin']:
            cur_iter = self.current_epoch * self.batch_num + batch_idx
            self.margin_scheduler.step(cur_iter)

        data, label = batch
        loss, acc = self(data, label)
        tqdm_dict = {"acc": acc}
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
            })
        return output

    def train_dataloader(self):
        frames_len = np.random.randint(self.hparams.min_frames, self.hparams.max_frames)
        print("\nChunk size is: ", frames_len)
        print("Augmentation Probability: ", self.hparams.aug_prob)
        print("Speed Perturb Mode: ", self.hparams.speed_perturb)
        print("Learning rate is: ", self.lr_scheduler.get_lr()[0])
        print("Margin is: ", self.loss.m)
        train_dataset = Train_Dataset(
                self.hparams.train_list_path, self.hparams.aug_prob, 
                self.hparams.speed_perturb, self.hparams.VTLP,
                warp_par=self.hparams.warp_par,
                speed_par=self.hparams.speed_par,
                musan_list_path=self.hparams.musan_list_path, 
                rirs_list_path=self.hparams.rirs_list_path,
                max_frames=frames_len, sample_rate=self.hparams.sample_rate)
        # train_sampler = Train_Sampler(train_dataset, self.hparams.nPerSpeaker,
        #         self.hparams.max_seg_per_spk, self.hparams.batch_size)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                # sampler=train_sampler,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                )
        return loader

    def eval_dataloader(self):
        eval_list = [[key, data] for key, data in self.eval_data.items()]
        print("number of eval: ", len(eval_list))

        test_dataset = Test_Dataset(data_list=eval_list, eval_frames=self.hparams.eval_frames, 
                                    num_eval=0, data_key_level=self.hparams.data_key_level)
        loader = DataLoader(test_dataset, num_workers=self.hparams.num_workers, batch_size=1)
        return loader

    def extract_embeddings(self):
        print("extract embeddings...")
        self.speaker_encoder.eval()
    
        # extract eval embeddings
        eval_ark = os.path.join(os.path.abspath(self.hparams.xvec_path), 'xvector.ark')
        eval_loader = self.eval_dataloader()
        self.extract(eval_loader, eval_ark)

    def extract(self, dataloader, embedding_ark):
        embedding_scp = embedding_ark[:-3] + 'scp'
        with torch.no_grad():
            with kaldiio.WriteHelper('ark,scp:' + embedding_ark + ',' + embedding_scp) as writer:
                for idx, (data, utt) in enumerate(tqdm(dataloader)):
                    utt = list(utt)[0]
                    data = data.permute(1, 0, 2).cuda()
                    embedding = self.extract_speaker_embedding(data)
                    embedding = torch.mean(embedding, axis=0)
                    embedding = embedding.cpu().detach().numpy()
                    writer(utt, embedding)

    def cosine_evaluate(self):
        eval_loader = self.eval_dataloader()
        index_mapping = {}
        eval_vectors = [[] for _ in range(len(eval_loader))]
        print("extract eval speaker embedding...")
        self.speaker_encoder.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(eval_loader)):
                data = data.permute(1, 0, 2).cuda()
                label = list(label)[0]
                index_mapping[label] = idx
                embedding = self.extract_speaker_embedding(data)
                embedding = torch.mean(embedding, axis=0)
                embedding = embedding.cpu().detach().numpy()
                eval_vectors[idx] = embedding
        eval_vectors = np.array(eval_vectors)
        print("start cosine scoring...")
        if self.hparams.apply_metric:
            eer, th, mindcf_e, mindcf_h = cosine_score(self.trials, index_mapping, eval_vectors, self.hparams.get('scores_path', ''))
            print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer * 100, mindcf_e, mindcf_h))
            self.log('cosine_eer', eer*100)
            self.log('minDCF(0.01)', mindcf_e)
            self.log('minDCF(0.001)', mindcf_h)
            return eer, th, mindcf_e, mindcf_h
        else:
            cosine_score(self.trials, index_mapping, eval_vectors, self.hparams.get('scores_path', ''), apply_metric=False)
            print("complete cosine scoring...")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        print("init {} optimizer with learning rate {}".format("Adam", self.lr_scheduler.get_lr()[0]))
        print("init Step lr_scheduler with step size {} and gamma {}".format(self.hparams.lr_step_size, self.hparams.lr_gamma))
        print('Warm-up epoch:', self.hparams.get('warm_up_epoch', 5))
        return [optimizer], [self.lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        warm_up_steps = self.hparams.get('warm_up_epoch', 5) * self.batch_num // self.hparams.get('nPerSpeaker', 1)
        if self.trainer.global_step < warm_up_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / warm_up_steps)
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

