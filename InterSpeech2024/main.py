#!/usr/bin/env python
# encoding: utf-8

import os
import fire
import yaml
import inspect
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from trainer import Model, model_evaluation
from trainer.utils import set_seed, load_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger


def cli_main(config='config.yaml', **kwargs):
    # configs
    with open(config) as conf:
        yaml_config = yaml.load(conf, Loader=yaml.FullLoader)
    configs = dict(yaml_config, **kwargs)

    set_seed(configs.get('seed', 1))

    model = Model(**configs)

    if configs.get('evaluate', False) is not True:
        torch.backends.cudnn.benchmark = True
        configs['default_root_dir'] = os.path.join(configs.get('exp_dir', 'exp'), 'checkpoints')
        if configs['auto_lr'] is True:
            configs['auto_lr_find'] = True
        lr_monitor = LearningRateMonitor(logging_interval='step')
        configs['callbacks'] = [model_evaluation(), lr_monitor]
        valid_kwargs = inspect.signature(Trainer.__init__).parameters
        args = dict((arg, configs[arg]) for arg in valid_kwargs if arg in configs)
        if configs.get('finetune', False):
            assert configs.get('checkpoint_path', None) is not None
            print('Fine-tuning using pretrained model: ', configs['checkpoint_path'])
            load_checkpoint(model=model, path=configs['checkpoint_path'], keep_loss_weight=configs.get('keep_loss_weight', True))
            trainer = Trainer(**args)
        elif configs.get('checkpoint_path', None) is not None:
            print('trainer load: ', configs['checkpoint_path'])
            trainer = Trainer(**args, resume_from_checkpoint=configs['checkpoint_path'])
        else:
            print('trainer from scratch')
            trainer = Trainer(**args)
        
        logger = TensorBoardLogger('logs/', name='my_model')
        trainer.logger = logger
        trainer.fit(model)
    else:
        assert configs.get('checkpoint_path', None) is not None and configs.get('eval_scp_path', None) is not None
        load_checkpoint(model=model, path=configs['checkpoint_path'], keep_loss_weight=False)
        print("initial parameter from pretrain model {}".format(configs['checkpoint_path']))
        model.cuda()
        model.eval()
        with torch.no_grad():
            if configs.get('extract', False):
                model.extract_embeddings()
            else:
                model.cosine_evaluate()

if __name__ == '__main__':  # pragma: no cover
    fire.Fire(cli_main)
