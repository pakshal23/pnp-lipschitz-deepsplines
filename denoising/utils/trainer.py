import sys
import os
sys.path.insert(0,"/home/bohra/pnp-lipschitz-deepsplines")
import torch
from torch.utils.data import DataLoader
from dataloader import dataset
from models import simple_cnn
import json
from torch.utils import tensorboard
from utils import metrics
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = dataset.BSD500(config['train_dataloader']['train_data_file'])
        val_dataset = dataset.BSD500(config['val_dataloader']['val_data_file'])

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["train_dataloader"]["batch_size"], shuffle=config["train_dataloader"]["shuffle"], num_workers=config["train_dataloader"]["num_workers"],drop_last=True)
        self.batch_size = config["train_dataloader"]["batch_size"]
        self.val_dataloader = DataLoader(val_dataset, batch_size=config["val_dataloader"]["batch_size"], shuffle=config["val_dataloader"]["shuffle"], num_workers=config["val_dataloader"]["num_workers"])

        print('Building the model')
        # Build the model
        self.model = simple_cnn.SimpleCNN(num_layers=config['net_params']['num_layers'], num_channels=config['net_params']['num_channels'], kernel_size=config['net_params']['kernel_size'], 
                                    padding=config['net_params']['padding'], bias=config['net_params']['bias'], spectral_norm=config['net_params']['spectral_norm'], **config['activation_fn_params'])

        self.model = self.model.to(device)
        print("Number of parameters in the model: ", self.model.get_num_params())

        # Set up finite-difference matrices for DeepSpline Lipschitz projection 
        if (config['training_options']['lipschitz_1_proj']):
            self.model.init_D(self.device)
        # Set up the optimizer
        self.set_optimization()
        self.epochs = config["training_options"]['epochs']
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        
        self.save_epoch = config["logging_info"]['save_epoch']
        self.epochs_per_val = config['logging_info']['epochs_per_val']

        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        #config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        config_save_path = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

       
    def set_optimization(self):
        optim_name = self.config["optimizer"]["type"]
        lr = self.config["optimizer"]["lr"]

        # optimizer
        params_iter = self.model.parameters()

        if optim_name == 'Adam':
            self.optimizer = torch.optim.Adam(params_iter, lr=lr)
        elif optim_name == 'SGD':
            self.optimizer = torch.optim.SGD(params_iter, lr=lr)
        else:
            raise ValueError('Need to provide a valid optimizer type')


    def train(self):
        
        for epoch in range(self.epochs+1):

            self.train_epoch(epoch)
            
            if epoch % self.epochs_per_val == 0:
                self.valid_epoch(epoch)
                
            # SAVE CHECKPOINT
            if epoch % self.save_epoch == 0:
                self.save_checkpoint(epoch)
        
        self.writer.flush()
        self.writer.close()

        
    def train_epoch(self, epoch):

        self.model.train()
        log = {}
        tbar = tqdm(self.train_dataloader)
        for batch_idx, data in enumerate(tbar):
            
            noisy_data = data + (self.sigma/255.0)*torch.randn(data.shape) 
            data, noisy_data = data.to(self.device), noisy_data.to(self.device)

            self.optimizer.zero_grad()
            
            output = (noisy_data + self.model(noisy_data))/2.0

            # data fidelity
            data_fidelity = (self.criterion(output, data))/(self.batch_size)
            
            # regularization
            regularization = torch.zeros_like(data_fidelity)
            if self.model.using_deepsplines and self.config['training_options']['lmbda'] > 0:
                regularization = self.config['training_options']['lmbda'] * self.model.TV2()

            total_loss = data_fidelity + regularization
            total_loss.backward()
            self.optimizer_step()

            #log['total_loss'] = total_loss.detach().cpu().item()
            #log['data_fidelity'] = data_fidelity.detach().cpu().item()
            #log['regularization'] = regularization.detach().cpu().item()

            log['total_loss'] = total_loss.item()
            log['data_fidelity'] = data_fidelity.item()
            log['regularization'] = regularization.item()
            
            self.wrt_step = (epoch) * len(self.train_dataloader) + batch_idx
            self.write_scalars_tb(log)

            tbar.set_description('TRAIN ({}) | TotalLoss {:.3f} |'.format(epoch, log['total_loss']))
        return 

    
    def optimizer_step(self):

        self.optimizer.step()
        
        # Do the projection step to constrain the Lipschitz constant to 1
        if self.config['training_options']['lipschitz_1_proj']:
            self.model.lipschitz_1_projection()


    def valid_epoch(self, epoch):
        
        self.model.eval()
        tbar = tqdm(self.val_dataloader)

        loss_val = 0.0
        psnr_val = 0.0
        ssim_val = 0.
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                noisy_data = data + (self.sigma/255.0)*torch.randn(data.shape)
                data, noisy_data = data.to(self.device), noisy_data.to(self.device)

                output = (noisy_data + self.model(noisy_data))/2.0
                loss = self.criterion(output, data)
                loss_val = loss_val + loss.cpu().item()

                clamped_output = torch.clamp(output, 0., 1.)
                psnr_val = psnr_val + metrics.batch_PSNR(clamped_output, data, 1.)
                ssim_val = ssim_val + metrics.batch_SSIM(clamped_output, data, 1.)

            # PRINT INFO
            loss_val = loss_val/len(self.val_dataloader)
            tbar.set_description('VAL ({}) | Loss: {:.3f} |'.format(epoch, loss_val))

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'Validation'
            self.writer.add_scalar(f'{self.wrt_mode}/Loss', loss_val, epoch)
            psnr_val = psnr_val/len(self.val_dataloader)
            ssim_val = ssim_val/len(self.val_dataloader)
            self.writer.add_scalar(f'{self.wrt_mode}/PSNR', psnr_val, epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/SSIM', ssim_val, epoch)

        return


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)


    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth'
        torch.save(state, filename)

