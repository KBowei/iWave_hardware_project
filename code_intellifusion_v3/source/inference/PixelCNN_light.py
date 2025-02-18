import torch
import torch.optim as optim
from torch.autograd import Variable
import math
import sys
from torch.nn import functional as F
import numpy as np

torch.use_deterministic_algorithms(True)

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = torch.logical_or(x >= 1e-6, g < 0.0)
        t = pass_through_if + 0.0

        return grad1 * t


class Distribution_for_entropy2(torch.nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2, self).__init__()

    def forward(self, x, p_dec):
        channel = p_dec.size()[1]
        if channel % 3 != 0:
            raise ValueError(
                "channel number must be multiple of 3")
        gauss_num = channel // 3
        temp = torch.chunk(p_dec, channel, dim=1)
        temp = list(temp)
        # keep the weight  summation of prob == 1
        probs = torch.cat(temp[gauss_num * 2:], dim=1)
        probs = F.softmax(probs, dim=1)

        # process the scale value to non-zero  如果为0，就设为最小值1*e-6
        for i in range(gauss_num, gauss_num * 2):
            temp[i] = torch.abs(temp[i])
            temp[i][temp[i] < 1e-6] = 1e-6

        gauss_list = []
        for i in range(gauss_num):
            gauss_list.append(torch.distributions.normal.Normal(temp[i], temp[i + gauss_num]))

        likelihood_list = []
        for i in range(gauss_num):
            likelihood_list.append(torch.abs(gauss_list[i].cdf(x + 0.5) - gauss_list[i].cdf(x - 0.5)))

        likelihoods = 0
        for i in range(gauss_num):
            likelihoods += probs[:, i:i + 1, :, :] * likelihood_list[i]

        return likelihoods


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class conv2drelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(conv2drelu, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.act = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class PixelCNN(torch.nn.Module):
    def __init__(self):
        super(PixelCNN, self).__init__()

        self.internal_channel = 128
        self.num_params = 9
        
        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1,
                                     padding=0)
        self.conv_pre2 = MaskedConv2d('A', in_channels=128, out_channels=self.internal_channel, kernel_size=3, stride=1,
                                      padding=0)
        self.conv1 = conv2drelu(in_channels=self.internal_channel, out_channels=self.internal_channel)
        self.conv2 = conv2drelu(in_channels=self.internal_channel, out_channels=self.internal_channel)
        self.conv_post = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1,
                                         stride=1, padding=0)

    def forward(self, x):      
        x = self.conv_pre(x)
        x = self.conv_pre2(x)

        x = self.conv1(x)
        x = self.conv2(x)

        params = self.conv_post(x)
        return params


class PixelCNN_Context(torch.nn.Module):
    def __init__(self, context_num):
        super(PixelCNN_Context, self).__init__()

        self.internal_channel = 128
        self.num_params = 9

        self.conv_pre = MaskedConv2d('A', in_channels=1, out_channels=self.internal_channel, kernel_size=3, stride=1,
                                     padding=0)
        self.conv_pre2 = MaskedConv2d('A', in_channels=128, out_channels=self.internal_channel, kernel_size=3, stride=1,
                                      padding=0)
        self.conv1 = conv2drelu(in_channels=self.internal_channel, out_channels=self.internal_channel)
        self.conv2 = conv2drelu(in_channels=self.internal_channel, out_channels=self.internal_channel)
        self.conv_post = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.num_params, kernel_size=1,
                                         stride=1, padding=0)
        self.conv_c = torch.nn.Conv2d(in_channels=context_num, out_channels=self.internal_channel, kernel_size=3,
                                      stride=1, padding=0)
        self.conv_c2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel,
                                       kernel_size=3, stride=1, padding=0)

    def forward(self, x0):
        x = x0[:, 0:1, :, :]
        context = x0[:, 1:, :, :]
        x = self.conv_pre(x)
        x = self.conv_pre2(x)

        context = self.conv_c(context)
        context = self.conv_c2(context)
        
        x = x + context
        x = self.conv1(x)
        x = self.conv2(x)
        params = self.conv_post(x)

        return params


if __name__ == '__main__':
    func = PixelCNN_Context(1)
    x = torch.randn(1, 1, 256, 256)
    c = torch.randn(1, 1, 256, 256)
    y = torch.cat([x, c])

    print(y.shape)
