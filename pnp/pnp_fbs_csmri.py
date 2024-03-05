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
import pylops_gpu
import math


######################## Functions for TV ##########################
class LinOpGrad():
    
    def __init__(self, sizein, device):
        
        self.H = sizein[0]
        self.W = sizein[1]
        self.device = device
        
        self.Dop_0 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=0, device=device, togpu=(True, True))
        self.Dop_1 = pylops_gpu.FirstDerivative(self.H*self.W, dims=(self.H, self.W), dir=1, device=device, togpu=(True, True))
    
    def apply(self, x):
        x = x.to(torch.float)
        with torch.no_grad():
            x = x.to(self.device)
            out_0 = self.Dop_0*(x.reshape(-1))
            out_1 = self.Dop_1*(x.reshape(-1))
            out = torch.cat([torch.reshape(out_0, (1, self.H, self.W)), torch.reshape(out_1, (1, self.H, self.W))], dim=0)
            #out = out.to(vol.device)
            
        return out
        
    def applyJacobianT(self, y):
        
        with torch.no_grad():
            y = y.to(self.device)
            out = torch.reshape(self.Dop_0.H*(y[0,...].view(-1)), (self.H, self.W)) + torch.reshape(self.Dop_1.H*(y[1,...].view(-1)), (self.H, self.W))
            #out = out.to(y.device)
            out = out.unsqueeze(0)
            out = out.unsqueeze(0)
            
        return out # 1 x 1 x K x K


def enforce_box_constraints(x, xmin, xmax):
    out = torch.clamp(x, min=xmin, max=xmax)    
    return out


class CostTV():
    def __init__(self, sizein, lamb, device):
        
        self.sizein = sizein
        self.lamb = lamb
        self.device = device
        #self.bounds = [-float('Inf'), float('Inf')]
        self.bounds = [0.0, float('Inf')]
        
        # Gradient operator
        self.D = LinOpGrad(self.sizein, self.device)
        
        # Parameters for FGP for the prox
        self.gam = 1.0/8
        self.num_iter = 100
        
    def apply(self, x):
        """x"""
        with torch.no_grad():
            out = torch.sum(torch.sqrt(torch.sum(torch.pow(self.D.apply(x), 2), dim=0)))
        return self.lamb*out
    
    def apply_all(self, x):
        """x"""
        return self.apply(x)
        
    def applyProx(self, u, alpha):
        
        with torch.no_grad():
            alpha = alpha*self.lamb
            # Initializations
            P = torch.zeros(2, self.sizein[0], self.sizein[1], device=u.device)
            F = torch.zeros(2, self.sizein[0], self.sizein[1], device=u.device)
            t = 1.0
            for kk in range(self.num_iter):
                #Pnew = F + (self.gam/(alpha))*self.D.apply(u - alpha*self.D.applyJacobianT(F))
                Pnew = F + (self.gam/(alpha))*self.D.apply(enforce_box_constraints(u - alpha*self.D.applyJacobianT(F), self.bounds[0], self.bounds[1]))
                tmp = torch.clamp(torch.sqrt(torch.sum(torch.pow(Pnew, 2), dim=0)), min=1.0)
                Pnew = Pnew/tmp.expand(2, self.sizein[0], self.sizein[1])
                
                tnew = (1 + math.sqrt(1 + 4*(t**2)))/2
                F = Pnew + (t - 1)/tnew*(Pnew - P)
                t = tnew
                P = Pnew
        
        return enforce_box_constraints(u - alpha*self.D.applyJacobianT(P), self.bounds[0], self.bounds[1])

##############################################################################

def pnp_fbs_csmri_(model, im_orig, mask, noise_sigma, device, **opts):

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
    noise = np.random.normal(loc=0, scale=noise_sigma / 2.0, size=(m, n, 2)).view(np.complex128)
    noise = np.squeeze(noise)

    y_clean = np.fft.fft2(im_orig, norm='backward') * mask
    y = y_clean + noise
    meas_snr = 10*np.log10(np.sum(np.square(np.absolute(y_clean)))/np.sum(np.square(np.absolute(y-y_clean))))
    print('Measurement SNR: ', meas_snr)

    x_init = np.real(np.fft.ifft2(y)) # zero fill

    # Power iterations to compute step-size
    #L = power_iteration(mask, m, n, 100)
    #alpha = 1/L
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
        x = np.real(x)

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

    if verbose:
        return x, inc, x_init, zero_fill_snr, snr
    else:
        return x, inc, x_init, zero_fill_snr


def parse_arguments():
    parser = argparse.ArgumentParser(description='PnP Reconstruction')
    parser.add_argument('--exp_name', default=None, type=str, help='name of the experiment')
    parser.add_argument('--img_file', default='CS_MRI', type=str, help='Path to the image file')
    parser.add_argument('--mask_file', default='CS_MRI', type=str, help='Path to the mask file')
    parser.add_argument("--noise_sigma", type=float, default=0, help="Noise level for the denoising model")
    parser.add_argument('--model_dir', default='final_exps', type=str, help='Path to the model directory')
    parser.add_argument('--device', default="cpu", type=str, help='device location')
    parser.add_argument('--mode', default="nn", type=str, help='Mode of operation')
    parser.add_argument("--sigma", type=float, default=5, help="Noise level for the denoising model")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Step size for FBS")
    parser.add_argument("--maxitr", type=int, default=100, help="Number of iterations")
    parser.add_argument("--tv_lamb", type=float, default=0.0, help="Reg param for TV")
    parser.add_argument("--verbose", type=int, default=1, help="Whether printing the info out")
    args = parser.parse_args()
    return args


def check_directory(experiment):
    if not os.path.exists("exp_results"):
        os.makedirs("exp_results")
    path = os.path.join("exp_results", "mri")
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, experiment)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def power_iteration(mask, m, n, num_iter):
    b_k = np.random.rand(m,n)

    for _ in range(num_iter):
        b_kl_forward = np.fft.fft2(b_k, norm='backward') * mask
        A_b_k = np.fft.ifft2(b_kl_forward*mask, norm='forward')  
        A_b_k_norm = np.linalg.norm(A_b_k.reshape(-1))
        b_k = A_b_k/A_b_k_norm

    b_kl_forward = np.fft.fft2(b_k, norm='backward') * mask
    A_b_k = np.fft.ifft2(b_kl_forward*mask, norm='forward')
    alpha = A_b_k[0,0]/b_k[0,0]

    return np.absolute(alpha)


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


if __name__ == '__main__':

    # ---- input arguments ----
    args = parse_arguments()
    # CONFIG -> assert if config is here
    config_file = args.model_dir + '/config.json'
    config = json.load(open(config_file))

    # ---- load the model ----
    model = simple_cnn.SimpleCNN(num_layers=config['net_params']['num_layers'], num_channels=config['net_params']['num_channels'], kernel_size=config['net_params']['kernel_size'], 
                                    padding=config['net_params']['padding'], bias=config['net_params']['bias'], spectral_norm=config['net_params']['spectral_norm'], **config['activation_fn_params'])

    device = args.device
    model_file = args.model_dir + '/checkpoint_200.pth'
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

    # create the output directory and return the path to it
    path = check_directory(args.exp_name)

    with torch.no_grad():
        # ---- load the ground truth ----
        im_orig = np.array(Image.open(args.img_file), np.float64)
        im_orig = im_orig/255.0
        
        # ---- load mask matrix ----
        mat = sio.loadmat(args.mask_file)
        mask = mat.get('Q1').astype(np.float64)

        # ---- set options -----
        opts = dict(sigma=args.sigma, maxitr=args.maxitr, alpha=args.alpha, verbose=args.verbose, mode=args.mode, tv_lamb=args.tv_lamb)

        # ---- plug and play !!! -----
        if args.verbose:
            x_out, inc, x_init, zero_fill_snr, snr = pnp_fbs_csmri_(model, im_orig, mask, args.noise_sigma, device, **opts)
        else:
            x_out, inc, x_init, zero_fill_snr = pnp_fbs_csmri_(model, im_orig, mask, args.noise_sigma, device, **opts)

        # ---- print result -----
        out_snr = psnr(x_out, im_orig)
        print('Plug-and-Play PNSR: ', out_snr)
        metrics = {"PSNR": np.round(snr, 8), "Zero fill PSNR": np.round(zero_fill_snr, 8), }

        with open(f'{path}/snr.txt', 'w') as f:
            for k, v in list(metrics.items()):
                f.write("%s\n" % (k + ':' + f'{v}'))

        # ---- save result -----
        fig, ax1 = plt.subplots()
        ax1.plot(inc, 'b-', linewidth=1)
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Increment', color='b')
        ax1.set_title("Increment curve")
        fig.savefig(f'{path}/inc.png')
        plt.show()

        if args.verbose:
            fig, ax1 = plt.subplots()
            ax1.plot(snr, 'b-', linewidth=1)
            ax1.set_xlabel('iteration')
            ax1.set_ylabel('PSNR', color='b')
            ax1.set_title("PSNR curve")
            fig.savefig(f'{path}/snr.png')
            plt.show()

        torch.save(torch.from_numpy(x_out), f'{path}/fbs.pt')
        torch.save(torch.from_numpy(x_init), f'{path}/ifft.pt')
        x_out = scale(x_out)
        x_init = scale(x_init)

        imageio.imwrite(f'{path}/fbs.jpg', x_out)
        imageio.imwrite(f'{path}/xinit.jpg', x_init)



