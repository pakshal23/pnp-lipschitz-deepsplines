import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractproperty, abstractmethod
from models.ds_modules.ds_utils import spline_grid_from_range


###########################################################################
class DeepBSpline_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size,
                save_memory):

        x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                            max=(grid.item() * (size // 2 - 1)))

        floored_x = torch.floor(x_clamped / grid)  # left coefficient
        fracs = x_clamped / grid - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_memory = save_memory

        if save_memory is False:
            ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        else:
            ctx.size = size
            ctx.save_for_backward(x, coefficients_vect, grid,
                                  zero_knot_indexes)

            # compute leftmost and rightmost slopes for linear extrapolations
            # outside B-spline range
            num_activations = x.size(1)
            coefficients = coefficients_vect.view(num_activations, size)
            leftmost_slope = (coefficients[:, 1] - coefficients[:, 0])\
                .div(grid).view(1, -1, 1, 1)
            rightmost_slope = (coefficients[:, -1] - coefficients[:, -2])\
                .div(grid).view(1, -1, 1, 1)

            # peform linear extrapolations outside B-spline range
            leftExtrapolations = (x.detach() + grid * (size // 2))\
                .clamp(max=0) * leftmost_slope
            rightExtrapolations = (x.detach() - grid * (size // 2 - 1))\
                .clamp(min=0) * rightmost_slope
            # linearExtrapolations is zero for inputs inside B-spline range
            linearExtrapolations = leftExtrapolations + rightExtrapolations

            # add linear extrapolations to B-spline expansion
            activation_output = activation_output + linearExtrapolations

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        save_memory = ctx.save_memory

        if save_memory is False:
            fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        else:
            size = ctx.size
            x, coefficients_vect, grid, zero_knot_indexes = ctx.saved_tensors

            # compute fracs and indexes again (do not save them in ctx)
            # to save memory
            x_clamped = x.clamp(min=-(grid.item() * (size // 2)),
                                max=(grid.item() * (size // 2 - 1)))

            floored_x = torch.floor(x_clamped / grid)  # left coefficient
            # distance to left coefficient
            fracs = x_clamped / grid - floored_x

            # This gives the indexes (in coefficients_vect) of the left
            # coefficients
            indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        if save_memory is True:
            # Add gradients from the linear extrapolations
            tmp1 = ((x.detach() + grid * (size // 2)).clamp(max=0)) / grid
            grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                                (-tmp1 * grad_out).view(-1))
            grad_coefficients_vect.scatter_add_(0,
                                                indexes.view(-1) + 1,
                                                (tmp1 * grad_out).view(-1))

            tmp2 = ((x.detach() - grid * (size // 2 - 1)).clamp(min=0)) / grid
            grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                                (-tmp2 * grad_out).view(-1))
            grad_coefficients_vect.scatter_add_(0,
                                                indexes.view(-1) + 1,
                                                (tmp2 * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

###################################################################################
            

class DeepBSpline(ABC, nn.Module):
    """ nn.Module for DeepBspline activation functions. """
    def __init__(self,
                 mode,
                 num_activations,
                 size=51,
                 range_=4,
                 grid=None,
                 init='leaky_relu',
                 save_memory = False, 
                 **kwargs):

        ######
        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')
        if int(size) % 2 == 0:
            raise TypeError('size should be an odd number.')

        if range_ is None:
            if grid is None:
                raise ValueError('One of the two args (range_ or grid) '
                                 'required.')
            elif float(grid) <= 0:
                raise TypeError('grid should be a positive float...')
        elif grid is not None:
            raise ValueError('range_ and grid should not be both set.')

        super().__init__()

        self.mode = mode
        self.size = int(size)
        self.num_activations = int(num_activations)
        self.init = init

        if range_ is None:
            self.grid = torch.Tensor([float(grid)])
        else:
            grid = spline_grid_from_range(size, range_)
            self.grid = torch.Tensor([grid])

        #####
        self.save_memory = bool(save_memory)
        self.init_zero_knot_indexes()

        self.D2_filter = torch.Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)

        #####
        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor  # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor)  # spline coefficients

        if self.init == 'leaky_relu':
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.1)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs()
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :],
                                                   lambd=0.5)
        else:
            raise ValueError('init should be in [leaky_relu, relu, even_odd].')

        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self._coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))


    @property
    def device(self):
        """
        Get the module's device (torch.device)

        Returns the device of the first found parameter.
        """
        return getattr(self, next(self.parameter_names())).device

    @property
    def grid_tensor(self):
        """
        Get locations of B-spline coefficients.

        Returns:
            grid_tensor (torch.Tensor):
                size: (num_activations, size)
        """
        grid_arange = torch.arange(-(self.size // 2),
                                   (self.size // 2) + 1).mul(self.grid)

        return grid_arange.expand((self.num_activations, self.size))

    @property
    def coefficients_vect(self):
        """ B-spline vectorized coefficients. """
        return self._coefficients_vect

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)

    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        D2_filter = self.D2_filter.to(device=self.coefficients.device)

        # F.conv1d():
        # out(i, 1, :) = D2_filter(1, 1, :) *conv* coefficients(i, 1, :)
        # out.size() = (num_activations, 1, filtered_activation_size)
        # after filtering, we remove the singleton dimension
        return F.conv1d(self.coefficients.unsqueeze(1), D2_filter).squeeze(1)

    @staticmethod
    def parameter_names():
        """ Yield names of the module parameters """
        yield 'coefficients_vect'

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size +
                                  (self.size // 2))

    def reshape_forward(self, input):
        """
        Reshape inputs for deepspline activation forward pass, depending on
        mode ('conv' or 'fc').
        """
        input_size = input.size()
        if self.mode == 'fc':
            if len(input_size) == 2:
                # one activation per conv channel
                # transform to 4D size (N, num_units=num_activations, 1, 1)
                x = input.view(*input_size, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f'input size is {len(input_size)}D '
                                 'but should be 2D or 4D...')
        else:
            assert len(input_size) == 4, \
                'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        return x

    def reshape_back(self, output, input_size):
        """
        Reshape back outputs after deepspline activation forward pass,
        depending on mode ('conv' or 'fc').
        """
        if self.mode == 'fc':
            # transform back to 2D size (N, num_units)
            output = output.view(*input_size)

        return output

    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = input.size()
        x = self.reshape_forward(input)

        assert x.size(1) == self.num_activations, \
            f'{input.size(1)} != {self.num_activations}.'

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        output = DeepBSpline_Func.apply(x, self.coefficients_vect, grid,
                                        zero_knot_indexes, self.size,
                                        self.save_memory)

        if self.save_memory is False:
            # Linear extrapolations:
            # f(x_left) = leftmost coeff value + \
            #               left_slope * (x - leftmost coeff)
            # f(x_right) = second rightmost coeff value + \
            #               right_slope * (x - second rightmost coeff)
            # where the first components of the sums (leftmost/second
            # rightmost coeff value) are taken into account in
            # DeepBspline_Func() and linearExtrapolations adds the rest.

            coefficients = self.coefficients
            leftmost_slope = (coefficients[:, 1] - coefficients[:, 0])\
                .div(grid).view(1, -1, 1, 1)
            rightmost_slope = (coefficients[:, -1] - coefficients[:, -2])\
                .div(grid).view(1, -1, 1, 1)

            # x.detach(): gradient w/ respect to x is already tracked in
            # DeepBSpline_Func
            leftExtrapolations = (x.detach() + grid * (self.size // 2))\
                .clamp(max=0) * leftmost_slope
            rightExtrapolations = (x.detach() - grid * (self.size // 2 - 1))\
                .clamp(min=0) * rightmost_slope
            # linearExtrapolations is zero for inputs inside B-spline range
            linearExtrapolations = leftExtrapolations + rightExtrapolations

            output = output + linearExtrapolations

        output = self.reshape_back(output, input_size)

        return output

    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)

    def init_D(self, device):
        """
        Setting up the finite-difference matrix D
        """
        self.D = torch.zeros([self.size-1, self.size], device=device)
        h = self.grid.item()
        for i in range(self.size-1):
            self.D[i, i] = -1.0/h
            self.D[i, i+1] = 1.0/h

        # Transpose of D
        self.DT = torch.transpose(self.D, 0, 1)

    def lipschitz_1_projection(self):
        """
        Project the deepspline coefficients to have a Lipschitz constant <= 1.
        The Lipschitz projection step is done by dividing the coefficients by the maximum absolute slope
        """
        with torch.no_grad():
            spline_coeffs = self.coefficients.data
            spline_slopes = torch.matmul(spline_coeffs, self.DT)
            div_vals = torch.max(torch.abs(spline_slopes), dim=1, keepdim=True)
            max_abs_slopes = div_vals[0]
            max_abs_slopes[max_abs_slopes < 1.0] = 1.0
            new_spline_coeffs = torch.div(spline_coeffs, max_abs_slopes)
            self.coefficients_vect.data = new_spline_coeffs.view(-1)
