import torch
import json
from dataloader.dataset import BSD500
import sys
sys.path.insert(0,"/home/bohra/pnp-lipschitz-deepsplines")
from models.simple_cnn import SimpleCNN
from utils import metrics

#model_dir = 'exps/'
model_dir = '/home/bohra/pnp-lipschitz-deepsplines/denoising/logs/trial_run_ds_final_2'
#model_dir = '/home/bohra/pnp-lipschitz-deepsplines/pnp/trained_denoisers/sigma_5/ds_sn/3'
model_num = 140

config = json.load(open(model_dir + '/config.json'))
saved_params = torch.load(model_dir + '/checkpoints/checkpoint_' + str(model_num) + '.pth')
#saved_params = torch.load(model_dir + '/checkpoint_' + str(model_num) + '.pth')

model = SimpleCNN(num_layers=config['net_params']['num_layers'], num_channels=config['net_params']['num_channels'], 
                  kernel_size=config['net_params']['kernel_size'], 
                padding=config['net_params']['padding'], bias=config['net_params']['bias'], spectral_norm=config['net_params']['spectral_norm'], **config['activation_fn_params'])

model.load_state_dict(saved_params['state_dict'])
model.eval()

test_dataset = BSD500('./data/test.h5')
sigma = 5.0
psnr, ssim = 0.0, 0.0

with torch.no_grad():
    for i in range(test_dataset.__len__()):
        print(i)
        data = test_dataset.__getitem__(i).unsqueeze(0)
        noisy_data = data + (sigma/255) * torch.randn(data.shape)
        output =  (model(noisy_data) + noisy_data)/2
        clamped_output = torch.clamp(output, 0., 1.)
        psnr = psnr + metrics.batch_PSNR(clamped_output, data, 1.)
        ssim = ssim + metrics.batch_SSIM(clamped_output, data, 1.)

print(psnr/test_dataset.__len__())
print(ssim/test_dataset.__len__())