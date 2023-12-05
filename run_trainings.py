from trainers.segmentation import *
import hydra
from omegaconf import DictConfig, OmegaConf
#from omegaconf.omegaconf import open_dict
import argparse

import torch
import torch.multiprocessing as mp
import numpy as np
import random


CONFIG_NAME = "base_cfg_infer"

def main_worker(rank, cfg):
    trainer = Segmentation_Trainer(rank, cfg)
    if rank == 0: print("Done...Starting Trainer...")
    trainer.train(verbose=True)

@hydra.main(version_base=None, config_path="./cfgs", config_name=CONFIG_NAME)
def launch(cfg : DictConfig) -> None:
    
    print("Loaded configs successfully...")
    print("===========================================================================================")
    print("===========================================================================================")
    print("Training configuration: ")
    print(OmegaConf.to_yaml(cfg))
    print("===========================================================================================")
    print("===========================================================================================")
    print("Setting up trainer...")

    torch.manual_seed(cfg.train_settings.set_manual_seed)
    np.random.seed(cfg.train_settings.set_manual_seed)
    random.seed(cfg.train_settings.set_manual_seed)

    cfg.train_settings.distributed = cfg.train_settings.distributed and torch.cuda.device_count() > 1

    if not cfg.train_settings.distributed:
        print('Normal Training')
        main_worker(0, cfg)
    else:
        print('Distributed Training')
        mp.spawn(main_worker, args=(cfg,), nprocs=torch.cuda.device_count())
    
    print("")

if __name__ == '__main__':
    
    launch()