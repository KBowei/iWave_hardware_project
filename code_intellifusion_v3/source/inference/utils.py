import Model
import argparse
import torch
from torchvision import transforms
from torch.autograd import Variable
import os
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import Quant
import copy
import time
import struct
import arithmetic_coding as ac

torch.use_deterministic_algorithms(True)

model_qps = range(28) # 13 MSE models + 14 perceptual models + 1 lossless model
model_lambdas = [0.4, 0.25, 0.16, 0.10, 0.0625, 0.039, 0.024, 0.015, 0.0095, 0.006, 0.0037, 0.002, 0.0012,
                 0.4, 0.25, 0.16, 0.10, 0.018, 0.012, 0.0075, 0.0048, 0.0032, 0.002, 0.00145, 0.0008, 0.00055, 0.00035,
                 9999
                 ]
qp_shifts=[
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 18, 14, 20, 13, 22],
[32, 30, 33, 34, 35, 36, 37],
[32, 31, 34, 30, 36, 29, 38],
[32, 30, 36, 28, 40, 26, 44],
[64, 60, 66, 56, 68, 70, 72],
[64, 62, 70, 58, 76, 54, 82],
[64, 58, 72, 52, 80, 46, 88],
[64, 56, 72, 48, 80, 40, 88],

[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[8, 7.5, 8.5, 7, 9, 6.5, 9.5],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 17, 14, 18, 13, 19],
[16, 15, 18, 14, 20, 13, 22],
[32, 30, 33, 34, 35, 36, 37],
[32, 31, 34, 30, 36, 29, 38],
[32, 30, 36, 28, 40, 26, 44],
[64, 60, 66, 56, 68, 52, 70],
[64, 62, 70, 60, 76, 58, 82],
[64, 58, 72, 52, 80, 46, 88],
[64, 56, 72, 48, 80, 40, 88],
[64, 56, 72, 48, 80, 40, 88],

[1]
]


def visual(x, path):
    size = x.size()
    toPIL = transforms.ToPILImage()
    pic = toPIL((x / 255.).squeeze(0).squeeze(0).cpu())
    pic.save(path)
    return


def subband2patch(x, code_block_size):
    b, c, h, w = x.shape
    x = x.unfold(2, code_block_size, code_block_size).unfold(3, code_block_size, code_block_size)
    x = x.contiguous().view(b, c, -1, code_block_size, code_block_size)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, code_block_size, code_block_size)
    return x


def patch2subband(x, code_block_size, h, w):
    b, c, h_p, w_p = x.shape
    n_patch_h = h // code_block_size
    n_patch_w = w // code_block_size
    n_patches = n_patch_h * n_patch_w
    bs = b // n_patches
    x = x.reshape(bs, n_patches, c, h_p, w_p)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs, c, n_patch_h, n_patch_w, h_p, w_p)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(bs, c, n_patch_h * h_p, n_patch_w * w_p)
    return x


def subband2patch_padding(x, h, w, stride, padding):
    size = x.size()
    x_tmp = x[:, :, 0:h, 0:w]
    for i in range(0, size[2]-2*padding, stride):
        for j in range(0, size[3]-2*padding, stride):
            x_tmp = torch.cat((x_tmp, x[:, :, i:i+h, j:j+w]), dim=0)
    return x_tmp[size[0]::, :, :, :]

   
# yuv and patches cat in batch dimension
def subbands2patch(LL_list, HL_list, LH_list, HH_list, code_block_size, y_padding_sub_h, y_padding_sub_w, uv_padding_sub_h, uv_padding_sub_w, trans_steps):
    bs = LL_list[0].shape[0]
    HL_list_new = []
    LH_list_new = []
    HH_list_new = []
    patch_num = []
    for c in range(3):
        if c == 0:
            paddings = (0, y_padding_sub_w[3], 0, y_padding_sub_h[3])
        else:
            paddings = (0, uv_padding_sub_w[3], 0, uv_padding_sub_h[3])

        LL_list[c] = F.pad(LL_list[c], paddings, "constant")
        LL_list[c] = subband2patch(LL_list[c], code_block_size)
        for i in range(trans_steps):
            j = trans_steps - 1 - i
            if c == 0:
                paddings = (0, y_padding_sub_w[j], 0, y_padding_sub_h[j])
            else:
                paddings = (0, uv_padding_sub_w[j], 0, uv_padding_sub_h[j])
            
            HL_list[c][j] = F.pad(HL_list[c][j], paddings, "constant")
            HL_list[c][j] = subband2patch(HL_list[c][j], code_block_size)
            
            LH_list[c][j] = F.pad(LH_list[c][j], paddings, "constant")
            LH_list[c][j] = subband2patch(LH_list[c][j], code_block_size)
            
            HH_list[c][j] = F.pad(HH_list[c][j], paddings, "constant")
            HH_list[c][j] = subband2patch(HH_list[c][j], code_block_size)
    
    # cat the YUV patches to the batch dimension to process in parallel
    LL = torch.cat(LL_list, dim=0)
    for i in range(trans_steps):
        patch_num_cur_level = []
        j = trans_steps - 1 - i 
        HL_list_new.append(torch.cat((HL_list[0][j], HL_list[1][j], HL_list[2][j]), dim=0))
        LH_list_new.append(torch.cat((LH_list[0][j], LH_list[1][j], LH_list[2][j]), dim=0))
        HH_list_new.append(torch.cat((HH_list[0][j], HH_list[1][j], HH_list[2][j]), dim=0))
        
        patch_num_cur_level.append(HL_list[0][j].shape[0] // bs)
        patch_num_cur_level.append(HL_list[1][j].shape[0] // bs)
        patch_num_cur_level.append(HL_list[2][j].shape[0] // bs)
        patch_num.append(patch_num_cur_level)
    return LL, HL_list_new, LH_list_new, HH_list_new, patch_num


def patch2subbands(LL, HL_list, LH_list, HH_list, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, code_block_size):
    trans_steps = len(patch_num)
    patch_num_cur_level = patch_num[0]
    y_cur_h = y_subband_h[trans_steps - 1]
    y_cur_w = y_subband_w[trans_steps - 1]
    uv_cur_h = uv_subband_h[trans_steps - 1]
    uv_cur_w = uv_subband_w[trans_steps - 1]
    y_cur_h_pad = (int(np.ceil(y_cur_h / code_block_size)) * code_block_size)
    y_cur_w_pad = (int(np.ceil(y_cur_w / code_block_size)) * code_block_size)
    uv_cur_h_pad = (int(np.ceil(uv_cur_h / code_block_size)) * code_block_size)
    uv_cur_w_pad = (int(np.ceil(uv_cur_w / code_block_size)) * code_block_size)
    
    LL_list = []
    HL_list_new = []
    LH_list_new = []
    HH_list_new = []

    y_channel_LL = patch2subband(LL[:patch_num_cur_level[0]], code_block_size, y_cur_h_pad, y_cur_w_pad)[:, :, :y_subband_h[trans_steps-1], :y_subband_w[trans_steps-1]]
    u_channel_LL = patch2subband(LL[patch_num_cur_level[0]:patch_num_cur_level[0]+patch_num_cur_level[1]], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[trans_steps-1], :uv_subband_w[trans_steps-1]]
    v_channel_LL = patch2subband(LL[patch_num_cur_level[0]+patch_num_cur_level[1]:], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[trans_steps-1], :uv_subband_w[trans_steps-1]]
    LL_list.extend([y_channel_LL, u_channel_LL, v_channel_LL])
    for i in range(trans_steps):
        j = trans_steps - 1 - i
        patch_num_cur_level = patch_num[i]
        y_cur_h = y_subband_h[j]
        y_cur_w = y_subband_w[j]
        uv_cur_h = uv_subband_h[j]
        uv_cur_w = uv_subband_w[j]
        y_cur_h_pad = (int(np.ceil(y_cur_h / code_block_size)) * code_block_size)
        y_cur_w_pad = (int(np.ceil(y_cur_w / code_block_size)) * code_block_size)
        uv_cur_h_pad = (int(np.ceil(uv_cur_h / code_block_size)) * code_block_size)
        uv_cur_w_pad = (int(np.ceil(uv_cur_w / code_block_size)) * code_block_size)
        tmp_HL_list = []
        tmp_LH_list = []
        tmp_HH_list = []

        y_channel_HL = patch2subband(HL_list[i][:patch_num_cur_level[0]], code_block_size, y_cur_h_pad, y_cur_w_pad)[:, :, :y_subband_h[j], :y_subband_w[j]]
        u_channel_HL = patch2subband(HL_list[i][patch_num_cur_level[0]:patch_num_cur_level[0]+patch_num_cur_level[1]], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        v_channel_HL = patch2subband(HL_list[i][patch_num_cur_level[0]+patch_num_cur_level[1]:], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        tmp_HL_list.extend([y_channel_HL, u_channel_HL, v_channel_HL])
        HL_list_new.append(tmp_HL_list)
        
        y_channel_LH = patch2subband(LH_list[i][:patch_num_cur_level[0]], code_block_size, y_cur_h_pad, y_cur_w_pad)[:, :, :y_subband_h[j], :y_subband_w[j]]
        u_channel_LH = patch2subband(LH_list[i][patch_num_cur_level[0]:patch_num_cur_level[0]+patch_num_cur_level[1]], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        v_channel_LH = patch2subband(LH_list[i][patch_num_cur_level[0]+patch_num_cur_level[1]:], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        tmp_LH_list.extend([y_channel_LH, u_channel_LH, v_channel_LH])
        LH_list_new.append(tmp_LH_list)

        y_channel_HH = patch2subband(HH_list[i][:patch_num_cur_level[0]], code_block_size, y_cur_h_pad, y_cur_w_pad)[:, :, :y_subband_h[j], :y_subband_w[j]]
        u_channel_HH = patch2subband(HH_list[i][patch_num_cur_level[0]:patch_num_cur_level[0]+patch_num_cur_level[1]], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        v_channel_HH = patch2subband(HH_list[i][patch_num_cur_level[0]+patch_num_cur_level[1]:], code_block_size, uv_cur_h_pad, uv_cur_w_pad)[:, :, :uv_subband_h[j], :uv_subband_w[j]]
        tmp_HH_list.extend([y_channel_HH, u_channel_HH, v_channel_HH])
        HH_list_new.append(tmp_HH_list)
        
        
    HL_list = list(zip(*HL_list_new))
    LH_list = list(zip(*LH_list_new))
    HH_list = list(zip(*HH_list_new))
    
    return LL_list, HL_list, LH_list, HH_list

def rgb2yuv_lossless(x):
    x = np.array(x, dtype=np.int32)

    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    b = x[:, :, 2:3]

    yuv = np.zeros_like(x, dtype=np.int32)

    Co = r - b
    tmp = b + np.right_shift(Co, 1)
    Cg = g - tmp
    Y = tmp + np.right_shift(Cg, 1)

    yuv[:, :, 0:1] = Y
    yuv[:, :, 1:2] = Co
    yuv[:, :, 2:3] = Cg

    return yuv


def yuv2rgb_lossless(x):
    x = np.array(x, dtype=np.int32)

    Y = x[:, :, 0:1]
    Co = x[:, :, 1:2]
    Cg = x[:, :, 2:3]

    rgb = np.zeros_like(x, dtype=np.int32)

    tmp = Y - np.right_shift(Cg, 1)
    g = Cg + tmp
    b = tmp - np.right_shift(Co, 1)
    r = b + Co

    rgb[:, :, 0:1] = r
    rgb[:, :, 1:2] = g
    rgb[:, :, 2:3] = b

    return rgb


def rgb2yuv(x):
    convert_mat = np.array([[0.299, 0.587, 0.114],
                            [-0.169, -0.331, 0.499],
                            [0.499, -0.418, -0.0813]], dtype=np.float32)

    y = x[:, 0:1, :, :] * convert_mat[0, 0] +\
        x[:, 1:2, :, :] * convert_mat[0, 1] +\
        x[:, 2:3, :, :] * convert_mat[0, 2]

    u = x[:, 0:1, :, :] * convert_mat[1, 0] +\
        x[:, 1:2, :, :] * convert_mat[1, 1] +\
        x[:, 2:3, :, :] * convert_mat[1, 2] + 128.

    v = x[:, 0:1, :, :] * convert_mat[2, 0] +\
        x[:, 1:2, :, :] * convert_mat[2, 1] +\
        x[:, 2:3, :, :] * convert_mat[2, 2] + 128.
    return torch.cat((y, u, v), dim=1)


def yuv2rgb(x):
    inverse_convert_mat = np.array([[1.0, 0.0, 1.402],
                                    [1.0, -0.344, -0.714],
                                    [1.0, 1.772, 0.0]], dtype=np.float32)
    r = x[:, 0:1, :, :] * inverse_convert_mat[0, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[0, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[0, 2]
    g = x[:, 0:1, :, :] * inverse_convert_mat[1, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[1, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[1, 2]
    b = x[:, 0:1, :, :] * inverse_convert_mat[2, 0] +\
        (x[:, 1:2, :, :] - 128.) * inverse_convert_mat[2, 1] +\
        (x[:, 2:3, :, :] - 128.) * inverse_convert_mat[2, 2]
    return torch.cat((r, g, b), dim=1)


def find_min_and_max(LL, HL_list, LH_list, HH_list):

    min_v = [[1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.],
             [1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.],
             [1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.]]
    max_v = [[-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.],
             [-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.],
             [-1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000., -1000000.]]

    for channel_idx in range(1):
        tmp = LL[channel_idx, 0, :, :]
        min_tmp = torch.min(tmp).item()
        max_tmp = torch.max(tmp).item()
        if min_tmp < min_v[channel_idx][0]:
            min_v[channel_idx][0] = min_tmp
        if max_tmp > max_v[channel_idx][0]:
            max_v[channel_idx][0] = max_tmp

        for s_j in range(4):
            s_i = 4 - 1 - s_j
            tmp = HL_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 1]:
                min_v[channel_idx][3 * s_i + 1] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 1]:
                max_v[channel_idx][3 * s_i + 1] = max_tmp

            tmp = LH_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 2]:
                min_v[channel_idx][3 * s_i + 2] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 2]:
                max_v[channel_idx][3 * s_i + 2] = max_tmp

            tmp = HH_list[s_i][channel_idx, 0, :, :]
            min_tmp = torch.min(tmp).item()
            max_tmp = torch.max(tmp).item()
            if min_tmp < min_v[channel_idx][3 * s_i + 3]:
                min_v[channel_idx][3 * s_i + 3] = min_tmp
            if max_tmp > max_v[channel_idx][3 * s_i + 3]:
                max_v[channel_idx][3 * s_i + 3] = max_tmp
    min_v = (np.array(min_v)).astype(np.int16)
    max_v = (np.array(max_v)).astype(np.int16)
    return min_v, max_v


# Check the number x // (code_block_size // 2) is in a left open and right closed interval, the left boundary is an even number and the right boundary is an adjacent odd number
def check_discard(x, code_block_size):
    if x % (code_block_size // 2) != 0:
        if (x // (code_block_size // 2)) % 2 == 0:
            return True
        else:
            return False
    else:
        if (x // (code_block_size // 2)) % 2 == 1:
            return True
        else:
            return False


def subband2patch_discard(LL, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, patch_num, code_block_size, trans_steps, idx):
    # LL shape: [num_blocks, 1, 2*code_block_size, 2*code_block_size]
    y_cur_h = y_subband_h[trans_steps - 1 - idx]
    y_cur_w = y_subband_w[trans_steps - 1 - idx]
    uv_cur_h = uv_subband_h[trans_steps - 1 - idx]
    uv_cur_w = uv_subband_w[trans_steps - 1 - idx]
    y_cur_h_pad = (int(np.ceil(y_cur_h / code_block_size)) * code_block_size)
    y_cur_w_pad = (int(np.ceil(y_cur_w / code_block_size)) * code_block_size)
    
    uv_cur_h_pad = (int(np.ceil(uv_cur_h / code_block_size)) * code_block_size)
    uv_cur_w_pad = (int(np.ceil(uv_cur_w / code_block_size)) * code_block_size)
    
    patch_num_cur_level = patch_num[idx]
    y_patch = LL[:patch_num_cur_level[0]]
    u_patch = LL[patch_num_cur_level[0]:patch_num_cur_level[0]+patch_num_cur_level[1]]
    v_patch = LL[patch_num_cur_level[0]+patch_num_cur_level[1]:]
    
    bs = y_patch.shape[0] // patch_num_cur_level[0]
    
    y_img = patch2subband(y_patch, code_block_size, y_cur_h_pad, y_cur_w_pad)
    y_patch = y_img.unfold(2, code_block_size, code_block_size).unfold(3, code_block_size, code_block_size)
    
    if check_discard(y_cur_h, code_block_size):
        y_patch = y_patch[:, :, :-1]
    if check_discard(y_cur_w, code_block_size):
        y_patch = y_patch[:, :, :, :-1]
    y_patch = y_patch.contiguous().view(bs, 1, -1, code_block_size, code_block_size)
    y_patch = y_patch.permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, code_block_size, code_block_size)    
    
    u_img = patch2subband(u_patch, code_block_size, uv_cur_h_pad, uv_cur_w_pad)
    v_img = patch2subband(v_patch, code_block_size, uv_cur_h_pad, uv_cur_w_pad)
    u_patch = u_img.unfold(2, code_block_size, code_block_size).unfold(3, code_block_size, code_block_size)
    v_patch = v_img.unfold(2, code_block_size, code_block_size).unfold(3, code_block_size, code_block_size)
    if check_discard(uv_cur_h, code_block_size):
        u_patch = u_patch[:, :, :-1]
        v_patch = v_patch[:, :, :-1]
    if check_discard(uv_cur_w, code_block_size):
        u_patch = u_patch[:, :, :, :-1]
        v_patch = v_patch[:, :, :, :-1]
    u_patch = u_patch.contiguous().view(bs, 1, -1, code_block_size, code_block_size)
    u_patch = u_patch.permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, code_block_size, code_block_size)
    v_patch = v_patch.contiguous().view(bs, 1, -1, code_block_size, code_block_size)
    v_patch = v_patch.permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, code_block_size, code_block_size)
    LL = torch.cat((y_patch, u_patch, v_patch), dim=0)
    return LL   


def get_entropy_params(LL, HL_list, LH_list, HH_list, model_coding_LL, model_coding_HL, model_coding_LH, model_coding_HH, model_transform_entropy, trans_steps, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w):
    # fix entropy inverse_transform scale 3.9914
    used_scale = 3.9914
    code_block_size = LL.shape[2]
    pad = torch.nn.ConstantPad2d(2, 0)
    params_LL = model_coding_LL(pad(LL))
    LL_tmp = LL * used_scale
    params_HL_list = []
    params_LH_list = []
    params_HH_list = []
    for i in range(trans_steps):
        # check the patches number is equal to the cur_level other subbands' patches number
        assert LL_tmp.shape[0] == HL_list[i].shape[0]
        j = trans_steps - 1 - i
        xxx = pad(torch.cat((HL_list[i], LL_tmp), 1))
        params_HL_list.append(model_coding_HL(xxx, j))
        HL_tmp = HL_list[i] * used_scale
        
        xxx = pad(torch.cat((LH_list[i], LL_tmp, HL_tmp), 1))
        params_LH_list.append(model_coding_LH(xxx, j))
        LH_tmp = LH_list[i] * used_scale
        
        xxx = pad(torch.cat((HH_list[i], LL_tmp, HL_tmp, LH_tmp), 1))
        params_HH_list.append(model_coding_HH(xxx, j))
        HH_tmp = HH_list[i] * used_scale
        
        # inverse_trans
        LL_tmp = model_transform_entropy.inverse_trans(LL_tmp, HL_tmp, LH_tmp, HH_tmp, j, ishalf=False)
        # subband2patch
        LL_tmp = subband2patch_discard(LL_tmp, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, patch_num, code_block_size, trans_steps, i)
    return params_LL, params_HL_list, params_LH_list, params_HH_list
        
    
def get_prob(params:torch.Tensor, yuv_low_bound:list, yuv_high_bound:list, gmm, freqs_resolution:int, idx:int):
    lower_bound = yuv_low_bound[idx]
    upper_bound = yuv_high_bound[idx]
    label = torch.arange(start=lower_bound, end=upper_bound+1, dtype=torch.int32).cuda()
    label = label.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    params = params.unsqueeze(4).repeat(1, 1, 1, 1, upper_bound - lower_bound + 1)
    prob = gmm(label, params)
    prob = prob + 1e-5
    prob = prob.squeeze(0).squeeze(0).view(-1, upper_bound - lower_bound + 1)
    prob = prob.cpu().data.numpy()
    prob = prob * freqs_resolution
    prob = prob.astype(np.int64)
    return prob
    
    
def arithmetic_coding(bin_path:str, code_block_bin_path:str, 
                      x:torch.Tensor, prob, shift_min:list, shift_max:list, min_v: list, patch_num: list, 
                      y_subband_h:list, y_subband_w:list, uv_subband_h:list, uv_subband_w:list, 
                      code_block_bytes_list:list, code_block_size:int, coded_coe_num:int, idx:int):
    
    trans_steps = len(patch_num)
    rlvn = 3 if idx == 0 else (idx-1) // 3
    patch_num_cur_level = patch_num[trans_steps-1-rlvn]
    cur_h = 0
    cur_w = 0
    cur_patch_num = 0
    cur_channel_patch_num = 0
    pad_num = 0
    bit_out = ac.CountingBitOutputStream(
        bit_out=ac.BitOutputStream(open(code_block_bin_path, "wb")))
    enc = ac.ArithmeticEncoder(bit_out)
    
    for sample_idx, prob_sample in enumerate(prob):
        # get the position in the patch
        if cur_h == code_block_size and cur_w == code_block_size:
            cur_h = 0
            cur_w = 0
            cur_patch_num += 1
            enc.finish()
            bit_out.close()
            code_block_bytes_list.append(bit_out.num_bits // 8)
            copy_binary_file(code_block_bin_path, bin_path)
            # open a new bin file
            bit_out = ac.CountingBitOutputStream(
            bit_out=ac.BitOutputStream(open(code_block_bin_path, "wb")))
            enc = ac.ArithmeticEncoder(bit_out)
        
        # get the yuv channel idx and height and width of the subband
        if cur_patch_num < patch_num_cur_level[0]:
            c = 0
            subband_h = y_subband_h[rlvn]
            subband_w = y_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = cur_patch_num
        elif cur_patch_num < (patch_num_cur_level[0]+patch_num_cur_level[1]):
            c = 1
            subband_h = uv_subband_h[rlvn]
            subband_w = uv_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = cur_patch_num - patch_num_cur_level[0]
        else:
            c = 2
            subband_h = uv_subband_h[rlvn]
            subband_w = uv_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = cur_patch_num - patch_num_cur_level[0] - patch_num_cur_level[1]
        # get the position in the subband
        coe_id = (cur_channel_patch_num // col_patch_num) * col_patch_num * code_block_size * code_block_size + cur_h * subband_w_pad + (cur_channel_patch_num % col_patch_num) * code_block_size + cur_w        
        # judge if the position is in the padding area, if not, encode the coe
        if (coe_id % subband_w_pad) < subband_w and (coe_id // subband_w_pad) < subband_h:
            if shift_min[c, idx] < shift_max[c, idx]:      
                freqs = ac.SimpleFrequencyTable(
                    prob_sample[shift_min[c, idx]:shift_max[c, idx] + 1])
                index = x[cur_patch_num, 0, cur_h, cur_w].cpu().data.numpy().astype(np.int16)
                data = index - min_v[c, idx]
                assert data >= 0
                try:
                    enc.write(freqs, data)
                except IndexError:
                    print(data)
                    print(shift_min[c, idx])
                    print(shift_max[c, idx])
            coded_coe_num = coded_coe_num + 1
        # get the position in the patch
        cur_w += 1
        if cur_w == code_block_size:
            cur_h += 1
            if cur_h != code_block_size:
                cur_w = 0
                
    enc.finish()          
    bit_out.close()
    code_block_bytes_list.append(bit_out.num_bits // 8)
    copy_binary_file(code_block_bin_path, bin_path)
    return coded_coe_num


def arithmetic_decoding(dec_list: list, x: torch.Tensor, prob, shift_min:list, shift_max:list, min_v: list, patch_num, 
                        y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, 
                        code_block_size:int, coded_coe_num:int, idx:int, h_i:int, w_i:int):
    index = []
    trans_steps = len(patch_num)
    rlvn = 3 if idx == 0 else (idx-1) // 3
    patch_num_cur_level = patch_num[trans_steps-1-rlvn]
    # len(prob) == sum(patch_num_cur_level)
    for sample_idx, prob_sample in enumerate(prob):
        # get the yuv channel idx and height and width of the subband
        if sample_idx < patch_num_cur_level[0]:
            c = 0
            subband_h = y_subband_h[rlvn]
            subband_w = y_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = sample_idx
        elif sample_idx < (patch_num_cur_level[0]+patch_num_cur_level[1]):
            c = 1
            subband_h = uv_subband_h[rlvn]
            subband_w = uv_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = sample_idx - patch_num_cur_level[0]
        else:
            c = 2
            subband_h = uv_subband_h[rlvn]
            subband_w = uv_subband_w[rlvn]
            subband_w_pad = (int(np.ceil(subband_w / code_block_size)) * code_block_size)
            col_patch_num = subband_w_pad // code_block_size
            cur_channel_patch_num = sample_idx - patch_num_cur_level[0] - patch_num_cur_level[1]
        # get the position in the subband
        coe_id = (cur_channel_patch_num // col_patch_num) * col_patch_num * code_block_size * code_block_size + h_i * subband_w_pad + (cur_channel_patch_num % col_patch_num) * code_block_size + w_i
        if (coe_id % subband_w_pad) < subband_w and (coe_id // subband_w_pad) < subband_h:
            if shift_min[c, idx] < shift_max[c, idx]:
                freqs = ac.SimpleFrequencyTable(
                    prob_sample[shift_min[c, idx]:shift_max[c, idx] + 1])
                dec_c = dec_list[sample_idx].read(freqs) + min_v[c, idx]
            else:
                dec_c = min_v[c, idx]
            coded_coe_num = coded_coe_num + 1
            index.append(dec_c)
        else:
            index.append(0)
    x[:, 0, h_i+2, w_i+2] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
    return coded_coe_num


def autoregressive_dec(bin_path: str, x: torch.Tensor, subband_entropy_model: torch.nn.Module, yuv_low_bound:list, yuv_high_bound:list, shift_max: list, shift_min: list, min_v: list, patch_num:list, code_block_bytes_list:list, 
                       y_subband_h:list, y_subband_w:list, uv_subband_h:list, uv_subband_w:list, 
                       gmm, freqs_resolution: int, code_block_size:int, coded_coe_num: int, idx: int, patch_idx: int, content_bits_start_position:int):
    h = x.size()[2]
    w = x.size()[3]
    pad = torch.nn.ConstantPad2d(2, 0)
    x = pad(x)
    lower_bound = yuv_low_bound[idx]
    upper_bound = yuv_high_bound[idx]
    label = torch.arange(start=lower_bound, end=upper_bound+1, dtype=torch.int32).cuda()
    label = label.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # get the decomposition level
    trans_steps = len(patch_num)
    rlvn = 3 if idx == 0 else (idx-1) // 3
    
    # init the decoder, a block bind a decoder
    total_patch_num = sum(patch_num[trans_steps - 1 - rlvn])
    dec_list = []
    bit_in_list = []
    for i in range(total_patch_num):
        offset = sum(code_block_bytes_list[:i+patch_idx])
        f = open(bin_path, "rb")
        f.seek(content_bits_start_position+offset)
        bit_in = ac.BitInputStream(f)
        dec = ac.ArithmeticDecoder(bit_in)
        bit_in_list.append(bit_in)
        dec_list.append(dec)
    patch_idx += total_patch_num
    for h_i in range(h):
        for w_i in range(w):
            cur_ct = x
            # deal with the LL subband
            if idx == 0:
                params = subband_entropy_model(cur_ct)[:, :, h_i:h_i+1, w_i:w_i+1]
            else:
                params = subband_entropy_model(cur_ct, rlvn)[:, :, h_i:h_i+1, w_i:w_i+1]
            params_ref = torch.load("/data/kangbw/models/hardware_project/output/enc_params/v3/params_LL.pt")[:, :, h_i:h_i+1, w_i:w_i+1]
            dif = torch.sum(torch.abs(params_ref - params))
            if dif != 0:
                print(f"dif: {dif} position: {h_i} {w_i}")
            prob = get_prob(params, yuv_low_bound, yuv_high_bound, gmm, freqs_resolution, idx)
            coded_coe_num = arithmetic_decoding(dec_list, x[:, 0:1], prob, shift_min, shift_max, min_v, patch_num, 
                                                y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, 
                                                code_block_size, coded_coe_num, idx, h_i, w_i)
    # close bin file
    for i in range(total_patch_num):
        bit_in = bit_in_list[i]
        bit_in.close()
    return x[:, 0:1, 2:-2, 2:-2], coded_coe_num, patch_idx
            
          
def save_params(params_LL, params_HL_list, params_LH_list, params_HH_list, trans_steps, path):
    if not os.path.exists(path):
            os.makedirs(path)
    torch.save(params_LL, path+f'params_LL.pt')
    for i in range(trans_steps):
        j = trans_steps-1-i
        torch.save(params_HL_list[i], path+f'params_HL_list_{j}.pt')
        torch.save(params_LH_list[i], path+f'params_LH_list_{j}.pt')
        torch.save(params_HH_list[i], path+f'params_HH_list_{j}.pt')
    torch.cuda.empty_cache()
    del params_LL
    del params_HL_list
    del params_LH_list
    del params_HH_list
    

# copy binary file
def copy_binary_file(src_path, dst_path):
    cur_size = os.path.getsize(dst_path)
    with open(src_path, 'rb') as src, open(dst_path, 'ab') as dst:
        while True:
            chunk = src.read(1024)
            if not chunk:
                break
            dst.write(chunk)
            
            
# write picture header to the bin file
def write_header(bin_path, isLossless, model_qp, qp_shift, isPostGAN, height, width, min_v, max_v, code_block_bytes_list):
    with open(bin_path, 'rb') as f:
       content_bits = f.read()
       
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('b', isLossless))
        f.write(struct.pack('b', model_qp))
        f.write(struct.pack('b', qp_shift))
        f.write(struct.pack('b', isPostGAN))
        f.write(struct.pack('h', height))
        f.write(struct.pack('h', width))
        for i in range(3):
            for j in range(13):
                f.write(struct.pack('h', min_v[i][j] + 6016))
                f.write(struct.pack('h', max_v[i][j] + 6016))
                
        f.write(struct.pack('h', len(code_block_bytes_list)))
        for i in range(len(code_block_bytes_list)):
            f.write(struct.pack('h', code_block_bytes_list[i]))
            
        f.write(content_bits)
        

def read_header(bin_path):
    with open(bin_path, 'rb') as f:
        # 读取并解包头部信息
        isLossless = struct.unpack('b', f.read(1))[0]
        model_qp = struct.unpack('b', f.read(1))[0]
        qp_shift = struct.unpack('b', f.read(1))[0]
        isPostGAN = struct.unpack('b', f.read(1))[0]
        height = struct.unpack('h', f.read(2))[0]
        width = struct.unpack('h', f.read(2))[0]

        # 读取 min_v 和 max_v
        min_v = []
        max_v = []
        for i in range(3):
            min_row = []
            max_row = []
            for j in range(13):
                min_value = struct.unpack('h', f.read(2))[0] - 6016
                max_value = struct.unpack('h', f.read(2))[0] - 6016
                min_row.append(min_value)
                max_row.append(max_value)
            min_v.append(min_row)
            max_v.append(max_row)
        min_v = np.array(min_v)
        max_v = np.array(max_v)  
          
        total_block_num = struct.unpack('h', f.read(2))[0]
        code_block_bytes_list = []
        for i in range(total_block_num):
            code_block_bytes_list.append(struct.unpack('h', f.read(2))[0])
        content_bits_start_position = f.tell()
    return isLossless, model_qp, qp_shift, isPostGAN, height, width, min_v, max_v, code_block_bytes_list, content_bits_start_position
            
    
    