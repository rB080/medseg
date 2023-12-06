from models.smp_models import smp_models
from models.unet import UNet
from models.ours import Attention_Guided_UNet
from dataloaders.load_data import *

from utils.segmentation_metrics import *
from utils.losses import *

import os.path as osp
import os
import signal

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm

import numpy as np



def handler(signum, frame): # handles timeouts and dumps overtime error
    raise Exception("Overtime Contingency!!")

def get_lr(optimizer, change_val=None): # fetches learning rate
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if change_val is not None: 
            assert change_val
            param_group['lr'] = change_val
        return lr

class Segmentation_Trainer(): # Trainer Class for managing training

    def __init__(self, rank, cfg): # takes rank and cfg (DictConfig) files to set up training driver...
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0) # checks if master process for verbosity during iterations
        self.device = cfg.train_settings.device # set device
        self.tot_gpus = torch.cuda.device_count()
        assert self.device in ['cpu', 'cuda', 'flex']
        if self.device == 'flex': 
            if torch.cuda.is_available(): 
                self.device = 'cuda'
                self.tot_gpus = torch.cuda.device_count() # get device count
            else: self.device = 'cpu'

        self.distributed = cfg.train_settings.distributed
        
        # Dataparallel utilities
        if cfg.train_settings.distributed:
            init_flag = False
            port_offset = 0
            while not init_flag: # flag to check timeout during init_process_group
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(50)
                try:
                    dist_url = "tcp://localhost:{}".format(str(29000 - port_offset)) #I'm not using this rn...
                    #os.environ['MASTER_ADDR'] = 'localhost'
                    #os.environ['MASTER_PORT'] = str(12355 - port_offset)
                    if self.is_master: print("Starting timer for 50 seconds to run init at port: ", str(29000 - port_offset))
                    dist.init_process_group(backend='nccl', init_method=dist_url,
                                            world_size=self.tot_gpus, rank=rank)
                    init_flag = True
                except Exception as exc:
                    print(exc)
                    print("Init Process Group timed out after 50 seconds...changing port offset...")
                    port_offset -= 1
                    init_flag = False
            signal.alarm(0)
            #del signal
            if self.is_master: print("Enabled Distributed training...")
        
        if self.device == 'cuda':
            torch.cuda.set_device(rank)
            self.device = torch.device('cuda', torch.cuda.current_device())

        # Initialize wandb for logging training runs
        if self.is_master and self.cfg.enable_wandb:
            os.environ['WANDB_DIR'] = os.path.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name)
            wandb.init(project='HVAE_Runs', config=dict(self.cfg))

    def log(self, k, v, epoch): # logs epoch data to wandb
        if self.cfg.enable_wandb:
            wandb.log({k: v}, step=epoch)
    
    def make_data(self): # defines datasets (train and test)
        if self.cfg.dataset == 'refuge':
            train_dataset = Refuge_Dataset(root_path=self.cfg.refuge_args.root, 
                                           split='train', 
                                           image_size=self.cfg.refuge_args.image_size, 
                                           mask_type=self.cfg.refuge_args.mask_type)
            test_dataset = Refuge_Dataset(root_path=self.cfg.refuge_args.root, 
                                          split='test', 
                                          image_size=self.cfg.refuge_args.image_size, 
                                          mask_type=self.cfg.refuge_args.mask_type)
        else:
            train_dataset = Isic_Dataset(root_path=self.cfg.isic_args.root, 
                                           split='train', 
                                           image_size=self.cfg.isic_args.image_size)
            test_dataset = Isic_Dataset(root_path=self.cfg.isic_args.root, 
                                          split='test', 
                                          image_size=self.cfg.isic_args.image_size)
        
        # sampler for distributed run
        if self.cfg.train_settings.distributed: 
            self.train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(test_dataset, shuffle=True)
        else: self.train_sampler, self.test_sampler = None, None

        # Loader definition
        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.dataloader.batch_size // self.tot_gpus, 
                                 drop_last=self.cfg.dataloader.drop_last, sampler=self.train_sampler,
                            shuffle=(True and self.train_sampler is None), num_workers=self.cfg.dataloader.num_workers // self.tot_gpus, 
                            pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.cfg.dataloader.batch_size // self.tot_gpus, 
                                 drop_last=self.cfg.dataloader.drop_last, sampler=self.test_sampler,
                            shuffle=False, num_workers=self.cfg.dataloader.num_workers // self.tot_gpus, 
                            pin_memory=True)
        
        
    def make_model(self): # Defines model instances for the trainer to run
        if self.cfg.model_type == 'unet':
            self.model = UNet(1)
        elif self.cfg.model_type[:3] == 'smp':
            self.model = smp_models(self.cfg.model_type[4:], 1)
        elif self.cfg.model_type == 'ours':
            self.model = Attention_Guided_UNet(1)
        
        # Model Distributed Dataparallel settings
        if self.cfg.train_settings.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model.cuda()
            self.model_ddp = DistributedDataParallel(self.model, device_ids=[self.rank])
        else: 
            self.model.to(self.device)
            self.model_ddp = self.model

        # Displays number of parameters
        if self.is_master: print("Number of Parameters: ", self.model.count_parameters())

    # function to set training variables
    def set_training(self):
        self.lr = self.cfg.train_settings.lr
        self.segmentation_loss = segmentation_loss()
        self.optimizer = torch.optim.Adam(self.model_ddp.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.cfg.train_settings.gamma)

    # function for forward prop
    def forward_prop(self, data, train=True):
        
        # selectively computes gradient if training
        if train:
            self.model_ddp.train()
        else:
            self.model_ddp.eval()
        
        # forward call
        if self.distributed: out = self.model_ddp.module.forward(data[0].to(self.device))
        else: out = self.model_ddp.forward(data[0].to(self.device))

        return out

    # function for backprop
    def backward_prop(self, out, gt):

        loss = self.segmentation_loss(gt, out, self.cfg.train_settings.loss_weights) # loss computation
        self.optimizer.zero_grad()
        loss.backward() # backward computation using autograd engine
        self.optimizer.step() # optimizer steps
        return loss

    def iter_dataset(self, train=True, verbose=True): # iterate over dataloader instance
        
        if verbose and self.is_master:
            if train: iterable = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            else: iterable = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        else:
            if train: iterable = enumerate(self.train_loader)
            else: iterable = enumerate(self.test_loader)

        iteration_loss = 0
        iteration_iou = 0
        iteration_dsc = 0
        iteration_acc = 0
        for idx, data_batch in iterable: # epoch iteration
            
            out = self.forward_prop(data=data_batch, train=train)
            if train:
                loss = self.backward_prop(out, data_batch[1].to(self.device)).sum().item()
            else:
                loss = self.segmentation_loss(out, data_batch[1].to(self.device), self.cfg.train_settings.loss_weights).sum().item()
            iteration_loss += loss

            # metric evaluation
            metrics = segmentation_metrics(out, data_batch[1].to(self.device))
            if self.is_master: 
                A, D, I = 'acc', 'dsc', 'iou'
                iterable.set_description(desc=f'train: acc={metrics[A]:.2f}, dsc={metrics[D]:.2f}, iou={metrics[I]:.2f}')
            iteration_iou += metrics['iou']
            iteration_dsc += metrics['dsc']
            iteration_acc += metrics['acc']

        self.scheduler.step()

        # compute mean of metrics
        if train:
            iteration_loss /= len(self.train_loader)
            iteration_iou /= len(self.train_loader)
            iteration_dsc /= len(self.train_loader)
            iteration_acc /= len(self.train_loader)
        else:
            iteration_loss /= len(self.test_loader)
            iteration_iou /= len(self.test_loader)
            iteration_dsc /= len(self.test_loader)
            iteration_acc /= len(self.test_loader)

        return iteration_loss, iteration_acc, iteration_dsc, iteration_iou
    
    def train(self, verbose=True): # primary training function
        
        # init train setup
        self.make_data()
        self.make_model()
        self.set_training()
        os.makedirs(osp.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name), exist_ok=True)

        ckpt_path = osp.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name, "segmentation_last.pth")
        if osp.isfile(ckpt_path) and not self.cfg.train_settings.clean_slate: # loads from checkpoint
            if self.is_master: print("Checkpoints found and clean_slate protocol not initiated...Loading models")
            self.load_checkpoint(save_type='last')
        
        best_loss = 10000

        for epoch in range(1, self.cfg.train_settings.epochs+1): # epoch set
            if self.is_master: print("===========================================================================================")
            if self.is_master: print("Starting to train Epoch: ", epoch, ", Learning rate: ", get_lr(self.optimizer))
            if self.is_master: print("===========================================================================================")
            loss, acc, dsc, iou = self.iter_dataset(train=True, verbose=verbose)
            if self.is_master: print("Average Epoch Loss: ", loss)
            if self.is_master: print("Average Epoch IOU: ", iou)
            if self.is_master: print("Average Epoch DSC: ", dsc)
            if self.is_master: print("Average Epoch ACC: ", acc)
            if self.is_master: print("===========================================================================================")

            if self.is_master: print("Saving checkpoint...") # checkpointing
            if loss < best_loss: 
                self.save_checkpoint('best')
                best_loss = loss
                self.log('best_loss', best_loss, epoch)
            if self.is_master: 
                self.log('loss', loss, epoch)
                self.log('acc', acc, epoch)
                self.log('dsc', dsc, epoch)
                self.log('iou', iou, epoch)
            self.save_checkpoint('last')

            # eval at every preset number of epochs
            if epoch % self.cfg.train_settings.eval_epochs == 0:
                if self.is_master: print("===========================================================================================")
                if self.is_master: print("Eval Run at Epoch: ", epoch)
                if self.is_master: print("===========================================================================================")
                loss, acc, dsc, iou = self.iter_dataset(train=False, verbose=verbose)
                if self.is_master: print("Average Epoch Loss: ", loss)
                if self.is_master: print("Average Epoch IOU: ", iou)
                if self.is_master: print("Average Epoch DSC: ", dsc)
                if self.is_master: print("Average Epoch ACC: ", acc)
                if self.is_master: print("===========================================================================================")
    
    def visualize(self): # eval and visualize trained models
        iterable = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        METRICS = {"acc": 0.0, "sen": 0.0, "pre": 0.0, "rec": 0.0, "iou": 0.0, "dsc": 0.0}
        collect_gt, collect_pred = [], []
        for idx, data_batch in iterable:
            
            out = self.forward_prop(data=data_batch, train=False)
            metrics = segmentation_metrics(out, data_batch[1].to(self.device))
            for k,v in metrics.items():
                METRICS[k] += v / len(self.test_loader)
            collect_gt.append(data_batch[1].detach().cpu().numpy())
            collect_pred.append(out.detach().cpu().numpy())
        print(METRICS)

        if self.cfg.dataset == 'refuge' or self.cfg.dataset == 'isic':
            plotROC(np.array(collect_gt).ravel(), 
                    np.array(collect_pred).ravel(), 
                    osp.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name, "ROC_plot_test.png"), 
                    self.cfg.model_type+"_"+self.cfg.dataset)


        

    def save_checkpoint(self, save_type='last'): # save function
        
        ckpt_path = osp.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name, "segmentation_{}.pth".format(save_type))
        ckpt_config = dict(self.cfg)
        save_dict = {}
        save_dict["cfgs"] = ckpt_config
        save_dict["model_state_dict"] = self.model.state_dict()
        save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        
        torch.save(save_dict, ckpt_path)
    
    def load_checkpoint(self, save_type='last'): # load function
        ckpt_path = osp.join(self.cfg.train_settings.checkpoint_dir, self.cfg.run_name, "segmentation_{}.pth".format(save_type))
        ckpt_config = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt_config["model_state_dict"])
        self.optimizer.load_state_dict(ckpt_config["optimizer_state_dict"])
        self.model.to(self.device)