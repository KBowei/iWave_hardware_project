import torch
from torch.nn import functional as F

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241] # bior4.4


class ResBlock(torch.nn.Module):
    def __init__(self, internal_channel=16):
        super(ResBlock, self).__init__()

        self.internal_channel = internal_channel
        # self.padding = torch.nn.ReflectionPad2d(1)
        self.padding = torch.nn.ZeroPad2d(1)
        self.conv1 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.internal_channel, out_channels=self.internal_channel, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(self.padding(x))
        out = self.relu(out)
        out = self.conv2(self.padding(out))
        return x + out


class P_block(torch.nn.Module):
    def __init__(self):
        super(P_block, self).__init__()
        # self.padding_reflect = torch.nn.ReflectionPad2d(1)
        self.padding = torch.nn.ZeroPad2d(1)
        self.conv_pre = torch.nn.Conv2d(1, 16, 3, 1, 0)  # 没有初始化
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.conv_post = torch.nn.Conv2d(16, 1, 3, 1, 0)

    def forward(self, x):
        x = self.conv_pre(self.padding(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_post(self.padding(x))
        return x


class learn_lifting97(torch.nn.Module):
    def __init__(self, trainable_set):
        super(learn_lifting97, self).__init__()
        if trainable_set:
            self.leran_wavelet_rate = 0.1
        else:
            self.leran_wavelet_rate = 0.0
        self.skip1 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip1.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[0]], [lifting_coeff[0]]]]]),
                                               requires_grad=False)
        self.p_block1 = P_block()

        self.skip2 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip2.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[1]], [lifting_coeff[1]], [0.0]]]]),
                                               requires_grad=False)
        self.p_block2 = P_block()

        self.skip3 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip3.weight = torch.nn.Parameter(torch.Tensor([[[[0.0], [lifting_coeff[2]], [lifting_coeff[2]]]]]),
                                               requires_grad=False)
        self.p_block3 = P_block()

        self.skip4 = torch.nn.Conv2d(1, 1, (3, 1), padding=0, bias=False)
        self.skip4.weight = torch.nn.Parameter(torch.Tensor([[[[lifting_coeff[3]], [lifting_coeff[3]], [0.0]]]]),
                                               requires_grad=False)
        self.p_block4 = P_block()

        self.n_h = 0.0
        self.n_l = 0.0

    def forward_trans(self, L, H):

        paddings = (0, 0, 1, 1)  

        tmp = F.pad(L, paddings, "constant", 0)
        skip1 = self.skip1(tmp)
        L_net = self.p_block1(skip1)
        H = H + skip1 + L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "constant", 0)
        skip2 = self.skip2(tmp)
        H_net = self.p_block2(skip2)
        L = L + skip2 + H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "constant", 0)
        skip3 = self.skip3(tmp)
        L_net = self.p_block3(skip3)
        H = H + skip3 + L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "constant", 0)
        skip4 = self.skip4(tmp)
        H_net = self.p_block4(skip4)
        L = L + skip4 + H_net * self.leran_wavelet_rate

        H = H * (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
        L = L * (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

        return L, H

    def inverse_trans(self, L, H):

        H = H / (lifting_coeff[4] + self.n_h * self.leran_wavelet_rate)
        L = L / (lifting_coeff[5] + self.n_l * self.leran_wavelet_rate)

        paddings = (0, 0, 1, 1)

        tmp = F.pad(H, paddings, "constant", 0)
        skip4 = self.skip4(tmp)
        H_net = self.p_block4(skip4)
        L = L - skip4 - H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "constant", 0)
        skip3 = self.skip3(tmp)
        L_net = self.p_block3(skip3)
        H = H - skip3 - L_net * self.leran_wavelet_rate

        tmp = F.pad(H, paddings, "constant", 0)
        skip2 = self.skip2(tmp)
        H_net = self.p_block2(skip2)
        L = L - skip2 - H_net * self.leran_wavelet_rate

        tmp = F.pad(L, paddings, "constant", 0)
        skip1 = self.skip1(tmp)
        L_net = self.p_block1(skip1)
        H = H - skip1 - L_net * self.leran_wavelet_rate
        return L, H

class Wavelet(torch.nn.Module):
    def __init__(self, trainable_set):
        super(Wavelet, self).__init__()
        
        self.lifting = learn_lifting97(trainable_set)
    def forward_trans(self, x):
        # transform for rows
        L = x[:,:,0::2,:]
        H = x[:,:,1::2,:]
        L, H = self.lifting.forward_trans(L, H)

        L = L.permute(0,1,3,2)
        LL = L[:,:,0::2,:]
        HL = L[:,:,1::2,:]
        LL, HL = self.lifting.forward_trans(LL, HL)
        LL = LL.permute(0,1,3,2)
        HL = HL.permute(0,1,3,2)

        H = H.permute(0,1,3,2)
        LH = H[:,:,0::2,:]
        HH = H[:,:,1::2,:]
        LH, HH = self.lifting.forward_trans(LH, HH)
        LH = LH.permute(0,1,3,2)
        HH = HH.permute(0,1,3,2)

        return LL, HL, LH, HH

    def inverse_trans(self, LL, HL, LH, HH):
        # LH = LH.permute(0, 1, 3, 2)
        # HH = HH.permute(0, 1, 3, 2)
        # H = torch.zeros(LH.size()[0], LH.size()[1], LH.size()[2] + HH.size()[2], LH.size()[3], device=LH.device)
        # LH, HH = self.lifting.inverse_trans(LH, HH)
        # H[:, :, 0::2, :] = LH
        # H[:, :, 1::2, :] = HH
        # H = H.permute(0, 1, 3, 2)

        # LL = LL.permute(0, 1, 3, 2)
        # HL = HL.permute(0, 1, 3, 2)
        # L = torch.zeros(LL.size()[0], LL.size()[1], LL.size()[2] + HL.size()[2], LL.size()[3], device=LH.device)
        # LL, HL = self.lifting.inverse_trans(LL, HL)
        # L[:, :, 0::2, :] = LL
        # L[:, :, 1::2, :] = HL
        # L = L.permute(0, 1, 3, 2)

        # L, H = self.lifting.inverse_trans(L, H)
        # x = torch.zeros(L.size()[0], L.size()[1], L.size()[2] + H.size()[2], L.size()[3], device=LH.device)
        # x[:, :, 0::2, :] = L
        # x[:, :, 1::2, :] = H
        bs = LL.shape[0]
        LH = LH.permute(0, 1, 3, 2)
        HH = HH.permute(0, 1, 3, 2)
        LH, HH = self.lifting.inverse_trans(LH, HH)
        H_cat = torch.cat((LH, HH), dim=3)
        H_reshape = H_cat.reshape(bs, 1, H_cat.shape[2] * 2, H_cat.shape[3] // 2)
        H = H_reshape.permute(0, 1, 3, 2)

        LL = LL.permute(0, 1, 3, 2)
        HL = HL.permute(0, 1, 3, 2)
        LL, HL = self.lifting.inverse_trans(LL, HL)
        L_cat = torch.cat((LL, HL), dim=3)
        L_reshape = L_cat.reshape(bs, 1, L_cat.shape[2] * 2, L_cat.shape[3] // 2)
        L = L_reshape.permute(0, 1, 3, 2)

        L, H = self.lifting.inverse_trans(L, H)
        x = torch.cat((L, H), dim=3)
        x = x.reshape(bs, 1, x.shape[2] * 2, x.shape[3] // 2)

        return x
