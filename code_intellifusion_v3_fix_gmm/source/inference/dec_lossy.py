import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F

import Model
from utils import (dec_split, model_lambdas, qp_shifts, yuv2rgb, subband2patch_discard, subbands2patch, patch2subbands, autoregressive_dec)
from PixelCNN_light import Distribution_for_entropy2
import os
from torchvision import transforms

torch.use_deterministic_algorithms(True)

def visual(x, path):
    size = x.size()
    toPIL = transforms.ToPILImage()
    pic = toPIL((x / 255.).squeeze(0).squeeze(0).cpu())
    pic.save(path)
    return

def get_visual_results(base_path, LL_list, HL_list, LH_list, HH_list):
    if os.path.exists(base_path) is False:
        os.makedirs(base_path)
    channel = ['y', 'u', 'v']
    h = LL_list[0].shape[2]
    w = LL_list[0].shape[3]
    h_list = [0]
    w_list = [0]
    trans_step = 4
    for i in range(trans_step + 1):
        h_list += [2**i * h]
        w_list += [2**i * w]
    for c in range(1):
        color = channel[c]
        tmp_path = os.path.join(base_path, color)
        result_image = torch.zeros(1, 1, h * 2**trans_step,
                                    w * 2**trans_step)
        result_image[:, :, :h, :w] = LL_list[c]
        for i in range(trans_step):
            j = trans_step - i - 1
            result_image[:, :, h_list[0]:h_list[i + 1],
                            w_list[i + 1]:w_list[i + 2]] = HL_list[c][i]
            result_image[:, :, h_list[i + 1]:h_list[i + 2],
                            w_list[0]:w_list[i + 1]] = LH_list[c][i]
            result_image[:, :, h_list[i + 1]:h_list[i + 2],
                            w_list[i + 1]:w_list[i + 2]] = HH_list[c][i]
        visual(result_image, tmp_path + '.png')
    return

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dec_binary(dec, bin_num):
    value = 0
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        dec_c = dec.read(freqs)
        value = value + (2**(bin_num-1-i))*dec_c
    return value


def dec_lossy(args, bin_path, freqs_resolution, logfile):

    trans_steps = 4
    code_block_size = args.code_block_size
    bin_name = os.path.basename(bin_path)[0:-4]
    with torch.no_grad():

        # isLossless, model_qp, qp_shift, isPostGAN, height_y, width_y, min_v, max_v, code_block_bytes_list, content_bits_start_position = read_header(bin_path)
        isLossless = False
        model_qp = 0
        qp_shift = 0
        isPostGAN = False
        height_y = 2160
        width_y = 3840
        all_scale = []
        for i in range(39):
            all_scale.append(10)
        infos = dec_split(bin_path, all_scale)
        
        init_scale = qp_shifts[model_qp][qp_shift]
        print(f'init_scale: {init_scale}')
        
        logfile.write(str(init_scale) + '\n')
        logfile.flush()

        # reload main model
        if model_qp > 12.5:  # 13-->26 Perceptual models
            checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_percep.pth')
        else:  # 0-->12 MSE models
            checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_mse.pth')
        print(f"Model_path: {os.path.join(args.model_path, str(model_lambdas[model_qp]) + '_mse.pth')}")
        all_part_dict = checkpoint['state_dict']

        models_dict = {}
        models_dict['transform'] = Model.Transform_aiWave(init_scale, isAffine=False)
        models_dict['coding_LL'] = Model.CodingLL()
        models_dict['coding_HL'] = Model.CodingHL()
        models_dict['coding_LH'] = Model.CodingLH()
        models_dict['coding_HH'] = Model.CodingHH()
        models_dict['transform_entropy'] = Model.Transform_aiWave(init_scale, isAffine=False)
        models_dict['post'] = Model.Post()

        if isPostGAN:
            assert (model_qp > 16.5)  # Perceptual models have PostGAN
            models_dict['postGAN'] = Model.PostGAN()

        models_dict_update = {}
        for key, model in models_dict.items():
            myparams_dict = model.state_dict()
            new_dict = all_part_dict     
            part_dict = {k: v for k, v in new_dict.items() if k in myparams_dict}
            myparams_dict.update(part_dict)
            model.load_state_dict(myparams_dict)
            if torch.cuda.is_available():
                model = model.cuda()
                # transform with fp16
                if key == 'transform':
                    model = model.half()
                model.eval()
            models_dict_update[key] = model
        models_dict.update(models_dict_update)
        
        gmm = Distribution_for_entropy2()
        
        print('===============Load pre-trained model succeed!=================')
        logfile.write('Load pre-trained model succeed!' + '\n')
        logfile.flush()

        
        LL_list = []
        pad_h_y = int(np.ceil(height_y / 16)) * 16 - height_y
        pad_w_y = int(np.ceil(width_y / 16)) * 16 - width_y
        y_subband_h = [(height_y + pad_h_y) // 2, (height_y + pad_h_y) // 4, (height_y + pad_h_y) // 8, (height_y + pad_h_y) // 16]
        y_subband_w = [(width_y + pad_w_y) // 2, (width_y + pad_w_y) // 4, (width_y + pad_w_y) // 8, (width_y + pad_w_y) // 16]
        y_padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in y_subband_h]
        y_padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in y_subband_w]
        LL_Y = torch.zeros(1, 1, y_subband_h[3], y_subband_w[3]).cuda()
        LL_list.append(LL_Y)
        
        height_uv = (pad_h_y + height_y) // 2
        width_uv = (pad_w_y + width_y) // 2
        pad_h_uv = int(np.ceil(height_uv / 16)) * 16 - height_uv
        pad_w_uv = int(np.ceil(width_uv / 16)) * 16 - width_uv
        uv_subband_h = [(height_uv + pad_h_uv) // 2, (height_uv + pad_h_uv) // 4, (height_uv + pad_h_uv) // 8, (height_uv + pad_h_uv) // 16]
        uv_subband_w = [(width_uv + pad_w_uv) // 2, (width_uv + pad_w_uv) // 4, (width_uv + pad_w_uv) // 8, (width_uv + pad_w_uv) // 16]
        uv_padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in uv_subband_h]
        uv_padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in uv_subband_w]
        LL_U = torch.zeros(1, 1, uv_subband_h[3], uv_subband_w[3]).cuda()
        LL_V = torch.zeros(1, 1, uv_subband_h[3], uv_subband_w[3]).cuda()
        LL_list.append(LL_U)
        LL_list.append(LL_V)
        
        HL_list = []
        LH_list = []
        HH_list = []
        tmp_HL_list = []
        tmp_LH_list = []
        tmp_HH_list = []
        down_scales = [2, 4, 8, 16]
        for i in range(trans_steps):
            tmp_HL_list.append(torch.zeros(1, 1, (height_y + pad_h_y) // down_scales[i],
                                       (width_y + pad_w_y) // down_scales[i]).cuda())
            tmp_LH_list.append(torch.zeros(1, 1, (height_y + pad_h_y) // down_scales[i],
                                       (width_y + pad_w_y) // down_scales[i]).cuda())
            tmp_HH_list.append(torch.zeros(1, 1, (height_y + pad_h_y) // down_scales[i],
                                       (width_y + pad_w_y) // down_scales[i]).cuda())
        HL_list.append(tmp_HL_list)
        LH_list.append(tmp_LH_list)
        HH_list.append(tmp_HH_list)
        # deal with uv
        for c in range(2):
            tmp_HL_list = []
            tmp_LH_list = []
            tmp_HH_list = []
            for i in range(trans_steps):
                tmp_HL_list.append(torch.zeros(1, 1, (height_uv + pad_h_uv) // down_scales[i],
                                        (width_uv + pad_w_uv) // down_scales[i]).cuda())
                tmp_LH_list.append(torch.zeros(1, 1, (height_uv + pad_h_uv) // down_scales[i],
                                        (width_uv + pad_w_uv) // down_scales[i]).cuda())
                tmp_HH_list.append(torch.zeros(1, 1, (height_uv + pad_h_uv) // down_scales[i],
                                        (width_uv + pad_w_uv) // down_scales[i]).cuda())
            HL_list.append(tmp_HL_list)
            LH_list.append(tmp_LH_list)
            HH_list.append(tmp_HH_list)  
        LL, HL_list, LH_list, HH_list, patch_num = subbands2patch(LL_list, HL_list, LH_list, HH_list, code_block_size, y_padding_sub_h, y_padding_sub_w, uv_padding_sub_h, uv_padding_sub_w, trans_steps)  
        print('========================Entropy Decoding=========================')
        coded_coe_num = 0
        saved_path = '/data/kangbw/models/hardware_project/output/decode/v3_fix_gmm/'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        # fix entropy inverse_transform scale 3.9914
        used_scale = 3.9914
        patch_idx = 0
        # decompress LL  
        print(f"patch_idx : {patch_idx}")
        LL, coded_coe_num, patch_idx = autoregressive_dec(infos, LL, models_dict['coding_LL'], patch_num, coded_coe_num, 0, patch_idx)
        print('LL decoded')
        # LL = torch.load(saved_path + f'LL.pt')
        torch.save(LL, saved_path + f'LL.pt')
        context = LL * used_scale
        for i in range(trans_steps):
            j = trans_steps - 1 - i
            # decompress HL
            print(f"patch_idx : {patch_idx}")
            HL, coded_coe_num, patch_idx = autoregressive_dec(infos, torch.cat((HL_list[i], context), dim=1), models_dict['coding_HL'], patch_num, coded_coe_num, 3*j+1, patch_idx)
            # HL = torch.load(saved_path + f'HL_list_{j}.pt')           
            HL_list[i] = HL
            print('HL' + str(j) + ' decoded')
            torch.save(HL, saved_path + f'HL_list_{j}.pt')
            HL = HL * used_scale

            # decompress LH
            print(f"patch_idx : {patch_idx}")
            LH, coded_coe_num, patch_idx = autoregressive_dec(infos, torch.cat((LH_list[i], context, HL), dim=1), models_dict['coding_LH'], patch_num, coded_coe_num, 3*j+2, patch_idx)
            # LH = torch.load(saved_path + f'LH_list_{j}.pt')
            LH_list[i] = LH
            print('LH' + str(j) + ' decoded')
            torch.save(LH, saved_path + f'LH_list_{j}.pt')
            LH = LH * used_scale

            # decompress HH
            print(f"patch_idx : {patch_idx}") 
            HH, coded_coe_num, patch_idx = autoregressive_dec(infos, torch.cat((HH_list[i], context, HL, LH), dim=1), models_dict['coding_HH'], patch_num, coded_coe_num, 3*j+3, patch_idx)
            # HH = torch.load(saved_path + f'HH_list_{j}.pt')
            HH_list[i] = HH
            print('HH' + str(j) + ' decoded')
            torch.save(HH, saved_path + f'HH_list_{j}.pt')
            HH = HH * used_scale

            context = models_dict['transform_entropy'].inverse_trans(context, HL, LH, HH, j)
            context = subband2patch_discard(context, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, patch_num, code_block_size, trans_steps, i)        
        # assert (coded_coe_num == (height_y + pad_h_y) * (width_y + pad_w_y) + 2 * (height_uv + pad_h_uv) * (width_uv + pad_w_uv))
        print('========================Inverse_transform=========================')
        recon_list = []
        LL_list, HL_list, LH_list, HH_list = patch2subbands(LL, HL_list, LH_list, HH_list, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, code_block_size)
        get_visual_results(args.recon_dir, LL_list, HL_list, LH_list, HH_list)
        # inverse transform has a different scale
        trans_used_scale = 5.9332
        for c in range(3):
            LL = LL_list[c] * trans_used_scale
            for i in range(trans_steps):
                j = trans_steps - 1 - i
                HL = HL_list[c][i] * trans_used_scale
                LH = LH_list[c][i] * trans_used_scale
                HH = HH_list[c][i] * trans_used_scale
                LL = models_dict['transform'].inverse_trans(LL, HL, LH, HH, j, ishalf=True)
            recon_list.append(LL.to(torch.float32))

        yuv444_u = F.interpolate(recon_list[1][:, :, 0:height_uv, 0:width_uv], scale_factor=2, mode='bicubic', align_corners=True).to('cuda')
        yuv444_v = F.interpolate(recon_list[2][:, :, 0:height_uv, 0:width_uv], scale_factor=2, mode='bicubic', align_corners=True).to('cuda')

        yuv444_y = recon_list[0][:, :, :height_y, :width_y]
        yuv444_u = yuv444_u[:, :, :height_y, :width_y]
        yuv444_v = yuv444_v[:, :, :height_y, :width_y]
        recon = torch.cat((yuv444_y, yuv444_u, yuv444_v), dim=0)
        recon = yuv2rgb(recon.permute(1, 0, 2, 3))
        recon = torch.clamp(torch.round(recon), 0., 255.)
        recon_save = recon[0, :, :, :]
        recon_save = recon_save.permute(1, 2, 0)
        recon_save = recon_save.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon_save, 'RGB')
        img.save(os.path.join(args.recon_dir, bin_name + 'before_post.png'))

        if isPostGAN:

            recon = models_dict['post'](recon)

            if height_y * width_y > 1080 * 1920:
                h_list = [0, height_y//2, height_y]
                w_list = [0, width_y//2, width_y]
                k_ = 2
            else:
                h_list = [0, height_y]
                w_list = [0, width_y]
                k_ = 1
            gan_rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height_y) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width_y) - w_list[_j + 1]
                    tmp = models_dict['postGAN'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                        w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    gan_rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                               -pad_start_h:tmp.size()[
                                                                                                                2] - pad_end_h,
                                                                                               -pad_start_w:tmp.size()[
                                                                                                                3] - pad_end_w]
            recon = gan_rgb_post
        else:

            h_list = [0, height_y//3, height_y//3*2, height_y]
            w_list = [0, width_y//3, width_y//3*2, width_y]
            k_ = 3
            rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height_y) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width_y) - w_list[_j + 1]
                    tmp = models_dict['post'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                        w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                               -pad_start_h:tmp.size()[
                                                                                                                2] - pad_end_h,
                                                                                               -pad_start_w:tmp.size()[
                                                                                                                3] - pad_end_w]
            recon = rgb_post

        recon = torch.clamp(torch.round(recon), 0., 255.)
        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        logfile.flush()
