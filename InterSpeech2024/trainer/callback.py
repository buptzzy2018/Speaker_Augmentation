#!/usr/bin/env python
# coding=utf-8

import os
import torch
from pytorch_lightning.callbacks import Callback


class model_evaluation(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pass


    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch = trainer.current_epoch + 1
        if epoch % pl_module.hparams.save_interval == 0 or epoch > pl_module.hparams.max_epochs-30:
            ckpt_name = "epoch=" + str(epoch) + ".ckpt"
            print('Saving checkpoint', ckpt_name, '...')
            save_path = os.path.join(trainer.default_root_dir, ckpt_name)
            trainer.save_checkpoint(save_path)

        if pl_module.hparams.eval_interval > 0 and epoch % pl_module.hparams.eval_interval == 0:
            pl_module.eval()
            with torch.no_grad():
                eer, th, mindcf_e, mindcf_h = pl_module.cosine_evaluate()
                with open(os.path.join(pl_module.hparams.exp_dir, 'epoch_score'), 'a') as f:
                    f.write('epoch={}\tCosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}\n'.format(epoch, eer * 100, mindcf_e, mindcf_h))
            pl_module.train()
