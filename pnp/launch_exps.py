import os
import numpy as np
import argparse
import json
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from models import simple_cnn
import imageio
from PIL import Image
import math



def scale(img):
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    #image = 255 * img
    image = 1 * img
    return image


def psnr(x, im_orig):
    xout = (x - np.min(x)) / (np.max(x) - np.min(x))
    norm1 = np.sum((np.absolute(im_orig)) ** 2)
    norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    psnr = 10 * np.log10(norm1 / norm2)
    return psnr


def pnp_fbs_csmri_(model, im_orig, mask, y, device, **opts):

    maxitr = opts.get('maxitr', 100)
    alpha = opts.get('alpha', 1e-5)
    sigma = opts.get('sigma', 5)
    verbose = opts.get('verbose',1)
    mode = opts.get('mode', 'nn')
    tv_lamb = opts.get('tv_lamb', 0.0)
    device = device
    inc = []
    if verbose:
        snr = []

    """ Initialization. """
    m, n = im_orig.shape
    x_init = np.fft.ifft2(y) # zero fill

    print('alpha: ', alpha)

    if (mode == 'tv'):
        print('tv reg param: ', tv_lamb)
        cost_tv = CostTV(im_orig.shape, tv_lamb, device)

    zero_fill_snr = psnr(x_init, im_orig)
    print('zero-fill PSNR:', zero_fill_snr)
    if verbose:
        snr.append(zero_fill_snr)

    x = np.copy(x_init)

    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)

        """ Update variables. """

        Hx = np.fft.fft2(x, norm='backward')*mask
        grad = np.fft.ifft2((Hx-y)*mask, norm='forward')
        x = x - alpha*grad
        # x = np.real( x )
        x = np.absolute(x)

        """ Denoising step. """

        if (mode == 'nn'):

            xtilde = np.copy(x)
            mintmp = np.min(xtilde)
            maxtmp = np.max(xtilde)
            xtilde = (xtilde - mintmp) / (maxtmp - mintmp)

            # the reason for the following scaling:
            # our denoisers are trained with "normalized images + noise"
            # so the scale should be 1 + O(sigma)

            scale_range = 1.0 + sigma / 255 / 2.0
            scale_shift = (1 - scale_range) / 2.0
            xtilde = xtilde * scale_range + scale_shift

            xtilde_torch = np.reshape(xtilde, (1, 1, m, n))
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).to(device, non_blocking=True)
            x = ((xtilde_torch + model(xtilde_torch))/2.0).cpu().numpy()
            x = np.reshape(x, (m, n))

            # scale and shift the denoised image back
            x = (x - scale_shift) / scale_range
            x = x * (maxtmp - mintmp) + mintmp

        elif (mode == 'tv'):
            
            xtilde_torch = np.reshape(np.copy(x), (1, 1, m, n))
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).to(device, non_blocking=True)
            x = cost_tv.applyProx(xtilde_torch, alpha)
            x = x.cpu().numpy()
            x = np.reshape(x, (m, n))

        elif (mode == 'id'):

            x = 1.0*x

        else:
            print('wrong mode')

        
        """ Monitoring. """
        if verbose:
            snr_tmp = psnr(x, im_orig)
            print("i: {}, \t psnr: {}".format(i + 1, snr_tmp))
            snr.append(snr_tmp)

        inc.append(np.sqrt(np.sum((np.absolute(x - xold)) ** 2)))

    x_init = np.real(x_init)
    if verbose:
        return x, inc, x_init, zero_fill_snr, snr
    else:
        return x, inc, x_init, zero_fill_snr


# PARSE THE ARGS
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--net_type', default='relu_sn', type=str,
                    help='Type of network')
parser.add_argument("--maxitr", type=int, default=100, help="Number of iterations")
parser.add_argument("--alpha", type=float, default=1e-5, help="Step size for FBS")
parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
parser.add_argument('--mode', default="nn", type=str, help='Mode of operation')
parser.add_argument("--tv_lamb", type=float, default=0.0, help="Reg param for TV")
parser.add_argument('-d', '--device', default="cpu", type=str,
                    help='device to use')
                    
args = parser.parse_args()

# PARAMETERS VALUES
img_type_list = ['Brain', 'Bust']
mask_type_list = ['Random', 'Radial', 'Cartesian']
noise_sigma_list = [10, 20, 30]

# Networks
layers_list = [5]
sigma_list = [5]

with torch.no_grad():
    for img_type in img_type_list:
        for mask_type in mask_type_list:
            for noise_sigma in noise_sigma_list:
            
                # Load image
                img_file = './mri_data/' + img_type + '.jpg'
                im_orig = np.array(Image.open(img_file), np.float64)
                im_orig = im_orig/255.0
                
                # Load mask
                mask_file = './mri_data/Q_' + mask_type + '30.mat'
                mat = sio.loadmat(mask_file)
                mask = mat.get('Q1').astype(np.float64)

                # Generate measurements
                m, n = im_orig.shape
                noise = np.random.normal(loc=0, scale=noise_sigma / 2.0, size=(m, n, 2)).view(np.complex128)
                noise = np.squeeze(noise)

                y_clean = np.fft.fft2(im_orig, norm='backward') * mask
                y = y_clean + noise
                meas_snr = 10*np.log10(np.sum(np.square(np.absolute(y_clean)))/np.sum(np.square(np.absolute(y-y_clean))))
                print('Measurement SNR: ', meas_snr)

                for sigma in sigma_list:
                    for layers in layers_list:

                        # ---- set options -----
                        opts = dict(sigma=sigma, maxitr=args.maxitr, alpha=args.alpha, verbose=args.verbose, mode=args.mode, tv_lamb=args.tv_lamb)
                        
                        out_path = './final_results/' + img_type + '/' + mask_type + '/' + 'noise_sigma_' + str(noise_sigma) + '/' +  args.net_type + '/' + str(layers) + '/sigma_' + str(sigma)
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)

                        print("###############################")
                        print(out_path)
                        print("###############################")

                        model_dir = './trained_denoisers/sigma_' + str(sigma) + '/' + args.net_type + '/' + str(layers)
                        config_file = model_dir + '/config.json'
                        config = json.load(open(config_file))

                        print(config)

                        # ---- load the model ----
                        model = simple_cnn.SimpleCNN(num_layers=config['net_params']['num_layers'], num_channels=config['net_params']['num_channels'], kernel_size=config['net_params']['kernel_size'], 
                                        padding=config['net_params']['padding'], bias=config['net_params']['bias'], spectral_norm=config['net_params']['spectral_norm'], **config['activation_fn_params'])
                
                        device = args.device
                        model_file = model_dir + '/checkpoint_200.pth'
                        checkpoint = torch.load(model_file, device)
                        if device == 'cpu':
                            for key in list(checkpoint['state_dict'].keys()):
                                if 'module.' in key:
                                    checkpoint['state_dict'][key.replace('module.', '')] = checkpoint['state_dict'][key]
                                    del checkpoint['state_dict'][key]

                        try:
                            model.load_state_dict(checkpoint['state_dict'], strict=True)
                        except Exception as e:
                            print(f'Some modules are missing: {e}')
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                        model.float()
                        model.eval()
                        if args.device != 'cpu':
                            model.to(device)


                        if args.verbose:
                            x_out, inc, x_init, zero_fill_snr, snr = pnp_fbs_csmri_(model, im_orig, mask, y, device, **opts)
                        else:
                            x_out, inc, x_init, zero_fill_snr = pnp_fbs_csmri_(model, im_orig, mask, y, device, **opts)

                        # ---- print result -----
                        out_snr = psnr(x_out, im_orig)
                        print('Plug-and-Play PNSR: ', out_snr)
                        metrics = {"PSNR": np.round(snr, 8), "Zero fill PSNR": np.round(zero_fill_snr, 8), }

                        with open(f'{out_path}/snr.txt', 'w') as f:
                            for k, v in list(metrics.items()):
                                f.write("%s\n" % (k + ':' + f'{v}'))

                        # ---- save result -----
                        fig, ax1 = plt.subplots()
                        ax1.plot(inc, 'b-', linewidth=1)
                        ax1.set_xlabel('iteration')
                        ax1.set_ylabel('Increment', color='b')
                        ax1.set_title("Increment curve")
                        fig.savefig(f'{out_path}/inc.png')
                        #plt.show()

                        if args.verbose:
                            fig, ax1 = plt.subplots()
                            ax1.plot(snr, 'b-', linewidth=1)
                            ax1.set_xlabel('iteration')
                            ax1.set_ylabel('PSNR', color='b')
                            ax1.set_title("PSNR curve")
                            fig.savefig(f'{out_path}/snr.png')
                            #plt.show()

                        torch.save(torch.from_numpy(x_out), f'{out_path}/fbs.pt')
                        torch.save(torch.from_numpy(x_init), f'{out_path}/ifft.pt')
                        x_out = scale(x_out)
                        x_init = scale(x_init)

                        imageio.imwrite(f'{out_path}/fbs.jpg', x_out)
                        imageio.imwrite(f'{out_path}/xinit.jpg', x_init)