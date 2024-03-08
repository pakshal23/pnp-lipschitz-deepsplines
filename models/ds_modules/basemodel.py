import torch
import torch.nn as nn
from torch import Tensor
from models.ds_modules.deepBspline import DeepBSpline


class BaseModel(nn.Module):
    def __init__(self,
                 activation_type=None,
                 num_classes=None,
                 spline_size=None,
                 spline_range=None,
                 spline_init=None,
                 save_memory=False,
                 **kwargs):
        """
        Args:
            ------ general -----------------------

            activation_type (str):
                'relu', 'leaky_relu', 'deepBspline'.
            num_classes (int):
                number of dataset classes.

            ------ deepspline --------------------

            spline_size (odd int):
                number of coefficients of spline grid;
                the number of knots is K = size - 2.
            spline_range (float):
                Defines the range of the B-spline expansion;
                B-splines range = [-spline_range, spline_range].
            spline_init (str):
                Function to initialize activations as (e.g. 'leaky_relu').
                Options: 'leaky_relu', 'relu' or 'even_odd';  
            save_memory (bool):
                If true, use a more memory efficient version (takes more time);
        """
        super().__init__()

        past_attr_names = dir(self)  # save attribute names

        # general attributes
        self.activation_type = activation_type
        self.num_classes = num_classes

        # deepspline attributes
        self.spline_init = spline_init
        self.spline_size = spline_size
        self.spline_range = spline_range
        self.save_memory = save_memory

        current_attr_names = dir(self)  # current attribute names
        # Get list of newly added attributes
        new_attr_names = list(set(current_attr_names) - set(past_attr_names))

        for attr_name in new_attr_names:
            # check that all the arguments were given (are not None).
            assert getattr(self, attr_name) is not None, \
                f'self.{attr_name} is None.'

        self.deepspline = None
        if self.activation_type == 'deepBspline':
            self.deepspline = DeepBSpline

    @property
    def device(self):
        """
        Get the network's device (torch.device). Returns the device of the first found parameter.
        """
        return next(self.parameters()).device

    ###########################################################################
    # Activation initialization

    def init_activation_list(self, activation_specs, **kwargs):
        """
        Initialize list of activation modules (deepspline or standard).

        Args:
            activation_specs (list):
                list of 2-tuples (mode[str], num_activations[int]);
                mode can be 'conv' (convolutional) or 'fc' (fully-connected);
                if mode='conv', num_activations = number of convolutional
                filters; if mode='fc', num_activations = number of units.
                len(activation_specs) = number of activation layers;
                e.g., [('conv', 64), ('fc', 100)].

        Returns:
            activations (nn.ModuleList)
        """
        assert isinstance(activation_specs, list), \
            f'activation_specs type: {type(activation_specs)}'

        if self.using_deepsplines:
            activations = nn.ModuleList()
            for mode, num_activations in activation_specs:
                activations.append(
                    self.deepspline(mode,
                                    num_activations,
                                    size=self.spline_size,
                                    range_=self.spline_range,
                                    init=self.spline_init,
                                    save_memory=self.save_memory))
        else:
            activations = self.init_standard_activations(activation_specs)

        return activations

    def init_activation(self, activation_specs, **kwargs):
        """
        Initialize a single activation module (deepspline or standard).

        Args:
            activation_specs (tuple):
                2-tuple (mode[str], num_activations[int]);
                mode can be 'conv' (convolutional) or 'fc' (fully-connected);
                if mode='conv', num_activations = number of convolutional
                filters; if mode='fc', num_activations = number of units.
                e.g. ('conv', 64).

        Returns:
            activation (nn.Module)
        """
        assert isinstance(activation_specs, tuple), \
            f'activation_specs type: {type(activation_specs)}'

        activation = self.init_activation_list([activation_specs], **kwargs)[0]

        return activation

    def init_standard_activations(self, activation_specs, **kwargs):
        """
        Initialize standard activation modules.

        Args:
            activation_specs :
                list of pairs (mode, num_channels/neurons);
                Only the length of this list matters for this function.

        Returns:
            activations (nn.ModuleList)
        """
        activations = nn.ModuleList()

        if self.activation_type == 'relu':
            relu = nn.ReLU()
            for i in range(len(activation_specs)):
                activations.append(relu)

        elif self.activation_type == 'leaky_relu':
            leaky_relu = nn.LeakyReLU()
            for i in range(len(activation_specs)):
                activations.append(leaky_relu)

        else:
            raise ValueError(f'{self.activation_type} '
                             'is not in relu family...')

        return activations

    def initialization(self, init_type='He'):
        """
        Initializes the network weights with 'He', 'Xavier', or a custom gaussian initialization.
        """
        assert init_type in ['He', 'Xavier', 'custom_normal']

        if init_type == 'He':
            if self.activation_type in ['leaky_relu', 'relu']:
                nonlinearity = self.activation_type
                slope_init = 0.01 if nonlinearity == 'leaky_relu' else 0.

            elif self.using_deepsplines and \
                    self.spline_init in ['leaky_relu', 'relu']:
                nonlinearity = self.spline_init
                slope_init = 0.01 if nonlinearity == 'leaky_relu' else 0.
            else:
                init_type = 'Xavier'  # overwrite init_type

        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                if init_type == 'Xavier':
                    nn.init.xavier_normal_(module.weight)

                elif init_type == 'custom_normal':
                    # custom Gauss(0, 0.05) weight initialization
                    module.weight.data.normal_(0, 0.05)
                    module.bias.data.zero_()

                else:  # He initialization
                    nn.init.kaiming_normal_(module.weight,
                                            a=slope_init,
                                            mode='fan_out',
                                            nonlinearity=nonlinearity)

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    ###########################################################################
    # Parameters

    def get_num_params(self):
        """
        Returns the total number of network parameters.
        """
        num_params = 0
        for param in self.parameters():
            num_params += torch.numel(param)

        return num_params

    def modules_deepspline(self):
        """
        Yields all deepspline modules in the network.
        """
        for module in self.modules():
            if isinstance(module, self.deepspline):
                yield module

    def named_parameters_no_deepspline(self, recurse=True):
        """
        Yields all named_parameters in the network,
        excepting deepspline parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            deepspline_param = False
            # get all deepspline parameters
            for param_name in self.deepspline.parameter_names():
                if name.endswith(param_name):
                    deepspline_param = True

            if deepspline_param is False:
                yield name, param

    def named_parameters_deepspline(self, recurse=True):
        """
        Yields all deepspline named_parameters in the network.
        """
        if not self.using_deepsplines:
            raise ValueError('Not using deepspline activations...')

        for name, param in self.named_parameters(recurse=recurse):
            deepspline_param = False
            # get all deepspline parameters
            for param_name in self.deepspline.parameter_names():
                if name.endswith(param_name):
                    deepspline_param = True

            if deepspline_param is True:
                yield name, param

    def parameters_no_deepspline(self):
        """
        Yields all parameters in the network,
        excepting deepspline parameters.
        """
        for name, param in self.named_parameters_no_deepspline(recurse=True):
            yield param

    def parameters_deepspline(self):
        """
        Yields all deepspline parameters in the network.
        """
        for name, param in self.named_parameters_deepspline(recurse=True):
            yield param

    def freeze_parameters(self):
        """
        Freezes the network (no gradient computations).
        """
        for param in self.parameters():
            param.requires_grad = False

    ##########################################################################
    # Deepsplines: regularization and sparsification

    @property
    def using_deepsplines(self):
        """
        True if using deepspline activations.
        """
        return (self.deepspline is not None)

    def l2sqsum_weights_biases(self):
        """
        Computes the sum of the l2 squared norm of the weights and biases
        of the network.

        Returns:
            l2sqsum (0d Tensor):
                l2sqsum = (sum(weights^2) + sum(biases^2))
        """
        l2sqsum = Tensor([0.]).to(self.device)

        for module in self.modules():
            if hasattr(module, 'weight') and \
                    isinstance(module.weight, nn.Parameter):
                l2sqsum = l2sqsum + module.weight.pow(2).sum()

            if hasattr(module, 'bias') and \
                    isinstance(module.bias, nn.Parameter):
                l2sqsum = l2sqsum + module.bias.pow(2).sum()

        return l2sqsum[0]  # 1-tap 1d tensor -> 0d tensor

    def TV2(self):
        """
        Computes the sum of the TV(2) (second-order total-variation)
        semi-norm of all deepspline activations in the network.

        Returns:
            tv2 (0d Tensor):
                tv2 = sum(TV(2))
        """
        tv2 = Tensor([0.]).to(self.device)

        for module in self.modules():
            if isinstance(module, self.deepspline):
                module_tv2 = module.totalVariation(mode='additive')
                tv2 = tv2 + module_tv2.norm(p=1)

        return tv2[0]  # 1-tap 1d tensor -> 0d tensor

    def init_D(self, device):
        """
        """
        for module in self.modules_deepspline():
            module.init_D(device)

    def lipschitz_1_projection(self):
        """
        Project all deepspline modules in the model to have a Lipschitz constant <= 1
        """
        for module in self.modules_deepspline():
            module.lipschitz_1_projection()

