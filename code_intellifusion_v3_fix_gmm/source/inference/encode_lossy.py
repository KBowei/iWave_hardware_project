import argparse
import copy
import os
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
import Model
from utils import (find_min_and_max, subband2patch, patch2subband, subbands2patch, patch2subbands, get_entropy_params,
                       model_lambdas, qp_shifts, rgb2yuv, yuv2rgb, save_params, write_bin)
import sys
from PixelCNN_light import Distribution_for_entropy2
import ac.arithmetic_coding as coder

torch.use_deterministic_algorithms(True)

def save_y_channel_to_txt(y_tensor, file_path):
    y_np = y_tensor.squeeze().cpu().numpy()
    np.savetxt(file_path, y_np, fmt='%d', delimiter=',')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def write_binary(enc, value, bin_num):
    bin_v = '{0:b}'.format(value).zfill(bin_num)
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        enc.write(freqs, int(bin_v[i]))

def enc_lossy(args):
    assert args.isLossless == 0

    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)

    logfile = open(os.path.join(args.log_dir, 'enc_log_{}.txt'.format(args.img_name[0:-4])), 'a')

    init_scale = qp_shifts[args.model_qp][args.qp_shift]
    print(f'init_scale: {init_scale}')
    logfile.write(str(init_scale) + '\n')
    logfile.flush()

    code_block_size = args.code_block_size

    bin_name = args.img_name[0:-4] + '_' + str(args.model_qp) + '_' + str(args.qp_shift)
    bin_path = os.path.join(args.bin_dir, bin_name + '.bin')
    code_block_bin_path = os.path.join(args.bin_dir, bin_name + '_code_block.bin')
    # create an empty bin file
    with open(bin_path, 'wb') as f:
        pass
    freqs_resolution = 1e6

    trans_steps = 4

    # choose different model according to the QP
    if args.model_qp > 12.5:  # 13-->26 Perceptual models
        checkpoint = torch.load(os.path.join(args.model_path, str(model_lambdas[args.model_qp]) + '_percep.pth'))
    else:  # 0-->12 MSE models
        checkpoint = torch.load(os.path.join(args.model_path, str(model_lambdas[args.model_qp]) + '_mse.pth'))
    print(f"Model_path: {os.path.join(args.model_path, str(model_lambdas[args.model_qp]) + '_mse.pth')}")

    all_part_dict = checkpoint['state_dict']

    models_dict = {}
    # init models
    models_dict['transform'] = Model.Transform_aiWave(init_scale, isAffine=False)
    models_dict['coding_LL'] = Model.CodingLL()
    models_dict['coding_HL'] = Model.CodingHL()
    models_dict['coding_LH'] = Model.CodingLH()
    models_dict['coding_HH'] = Model.CodingHH()
    models_dict['transform_entropy'] = Model.Transform_aiWave(init_scale, isAffine=False)
    models_dict['post'] = Model.Post()
    if args.isPostGAN:
        freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int16))
        assert (args.model_qp > 16.5)  # Perceptual models have PostGAN
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
    
    print('===============Load pre-trained model succeed!=================')
    logfile.write('Load pre-trained model succeed!' + '\n')
    logfile.flush()

    img_path = os.path.join(args.input_dir, args.img_name)

    with torch.no_grad():
        start = time.time()
        print(f"Input_image_path: {img_path}")
        logfile.write(img_path + '\n')
        logfile.flush()

        assert img_path[-3:] == 'yuv'
        # fixed height and width
        width = 3840
        height = 2160
        
        frame_size = width * height * 3 // 2  # NV12
        with open(img_path, 'rb') as f:
            yuv_data = f.read(frame_size)

        y = yuv_data[:width * height]  # Y data
        uv = yuv_data[width * height:]  # UV data

        y_np = np.frombuffer(y, dtype=np.uint8).reshape(height, width)
        uv_np = np.frombuffer(uv, dtype=np.uint8).reshape(height // 2, width)

        img_Y = torch.from_numpy(y_np.copy()).unsqueeze(0).unsqueeze(0).float().cuda()

        yuv420_u = torch.from_numpy(uv_np[:, ::2].copy()).unsqueeze(0).unsqueeze(0).float().cuda()
        yuv420_v = torch.from_numpy(uv_np[:, 1::2].copy()).unsqueeze(0).unsqueeze(0).float().cuda()

        yuv444_uo = F.interpolate(yuv420_u, scale_factor=2, mode='bicubic', align_corners=True).to('cuda')
        yuv444_vo = F.interpolate(yuv420_v, scale_factor=2, mode='bicubic', align_corners=True).to('cuda')
        ori = torch.cat((img_Y, yuv444_uo, yuv444_vo), dim=0)
        ori = yuv2rgb(ori.permute(1, 0, 2, 3))
        ori = ori[:, :, 0:height, 0:width]
        original_img = ori.cpu()
        original_y = img_Y.cpu()
        input0 = original_img[0, :, :, :]
        input0 = input0.permute(1, 2, 0)
        input0 = input0.data.numpy()
        input0 = np.clip(input0, 0, 255).astype(np.uint8)
        img = Image.fromarray(input0, 'RGB')
        img.save(os.path.join(args.recon_dir, 'input.png'))

        pad_h = int(np.ceil(height / 16)) * 16 - height
        pad_w = int(np.ceil(width / 16)) * 16 - width
        paddings = (0, pad_w, 0, pad_h)
        img_Y = F.pad(img_Y, paddings, 'replicate')
        y_subband_h = [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        y_subband_w = [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]
        # y_padding_sub_h = [(int(2**(trans_steps-1-i) * np.ceil(y_subband_h[3] / code_block_size)) * code_block_size - tmp) for i, tmp in enumerate(y_subband_h)]
        # y_padding_sub_w = [(int(2**(trans_steps-1-i) * np.ceil(y_subband_w[3] / code_block_size)) * code_block_size - tmp) for i, tmp in enumerate(y_subband_w)]
        y_padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in y_subband_h]
        y_padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in y_subband_w]

        height_UV = (pad_h + height) // 2
        width_UV = (pad_w + width) // 2
        pad_h_UV = int(np.ceil(height_UV / 16)) * 16 - height_UV
        pad_w_UV = int(np.ceil(width_UV / 16)) * 16 - width_UV
        paddings_UV = (0, pad_w_UV, 0, pad_h_UV)
        yuv420_u = F.pad(yuv420_u, paddings_UV, 'replicate')
        yuv420_v = F.pad(yuv420_v, paddings_UV, 'replicate')
        uv_subband_h = [(height_UV + pad_h_UV) // 2, (height_UV + pad_h_UV) // 4, (height_UV + pad_h_UV) // 8, (height_UV + pad_h_UV) // 16]
        uv_subband_w = [(width_UV + pad_w_UV) // 2, (width_UV + pad_w_UV) // 4, (width_UV + pad_w_UV) // 8, (width_UV + pad_w_UV) // 16]
        # uv_padding_sub_h = [(int(2**(trans_steps-1-i) * np.ceil(uv_subband_h[3] / code_block_size)) * code_block_size - tmp) for i, tmp in enumerate(uv_subband_h)]
        # uv_padding_sub_w = [(int(2**(trans_steps-1-i) * np.ceil(uv_subband_w[3] / code_block_size)) * code_block_size - tmp) for i, tmp in enumerate(uv_subband_w)]
        uv_padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in uv_subband_h]
        uv_padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in uv_subband_w]
        
        input_img_Y = to_variable(img_Y)
        input_img_U = to_variable(yuv420_u)
        input_img_V = to_variable(yuv420_v)

        index_YUV = -1
        # storage the encoded coe
        LL_list = []
        HL_list = []
        LH_list = []
        HH_list = []
        # process the Y, U, V channel respectively by the same transform model
        for i_YUV in [img_Y, yuv420_u, yuv420_v]:
            index_YUV += 1
            input_img_v = to_variable(i_YUV).half()
            # get the subbands of the image using the transform model which runs on GPU/NPU...
            tmp_LL, tmp_HL_list, tmp_LH_list, tmp_HH_list, trans_used_scale = models_dict['transform'].forward_trans(input_img_v)
            LL_list.append(tmp_LL)
            HL_list.append(tmp_HL_list)
            LH_list.append(tmp_LH_list)
            HH_list.append(tmp_HH_list)
        min_vy, max_vy = find_min_and_max(LL_list[0], HL_list[0], LH_list[0], HH_list[0])
        min_vu, max_vu = find_min_and_max(LL_list[1], HL_list[1], LH_list[1], HH_list[1])
        min_vv, max_vv = find_min_and_max(LL_list[2], HL_list[2], LH_list[2], HH_list[2])

        min_v = np.array([min_vy[0], min_vu[0], min_vv[0]], dtype=np.int16)
        max_v = np.array([max_vy[0], max_vu[0], max_vv[0]], dtype=np.int16)

        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound
        print(shift_min)
        print(shift_max)
        # subbands2patch for accelarate
        LL, HL_list, LH_list, HH_list, patch_num = subbands2patch(LL_list, HL_list, LH_list, HH_list, code_block_size, y_padding_sub_h, y_padding_sub_w, uv_padding_sub_h, uv_padding_sub_w, trans_steps)
        # LL shape[20, 1, 64, 64] HL_list length=4 shape: [20, 1, 64, 64], [64, 1, 64, 64], [215, 1, 64, 64], [780, 1, 64, 64]
        print("===============Get the subbands of the image succeed!=================")
        save_coefficient_path = '/data/kangbw/models/hardware_project/output/enc_coefficient/v3_fix_gmm/'
        if not os.path.exists(save_coefficient_path):
            os.makedirs(save_coefficient_path)
        torch.save(LL, save_coefficient_path + f'LL.pt')
        for i in range(trans_steps):
            j = trans_steps - 1 - i
            torch.save(HL_list[i], save_coefficient_path + f'HL_list_{j}.pt')
            torch.save(LH_list[i], save_coefficient_path + f'LH_list_{j}.pt')
            torch.save(HH_list[i], save_coefficient_path + f'HH_list_{j}.pt')
            
        # get the entropy params of the subbands using the entropy model which runs on GPU/FPGA...
        params_LL, params_HL_list, params_LH_list, params_HH_list = get_entropy_params(LL, HL_list, LH_list, HH_list, models_dict['coding_LL'], models_dict['coding_HL'], models_dict['coding_LH'], models_dict['coding_HH'], models_dict['transform_entropy'], trans_steps, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w)
        save_params_path = '/data/kangbw/models/hardware_project/output/enc_params/v3_fix_gmm/'
        save_params(params_LL, params_HL_list, params_LH_list, params_HH_list, trans_steps, path=save_params_path)
        params_LL, params_HL_list, params_LH_list, params_HH_list = patch2subbands(params_LL, params_HL_list, params_LH_list, params_HH_list, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, code_block_size)
        LL_list, HL_list, LH_list, HH_list = patch2subbands(LL, HL_list, LH_list, HH_list, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, code_block_size)
        all_gmm_list = []
        all_subband_list = []
        all_scale = []
        for i in range(39):
            all_scale.append(10)
        for c in range(3):
            all_subband_list.append(LL_list[c].flatten().cpu().numpy().astype(np.int16).tolist())
            all_gmm_list.append((params_LL[c] * 10).permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0).cpu().numpy().astype(np.int).tolist())
        
        for i in range(trans_steps):
            for c in range(3):
                all_subband_list.append(HL_list[c][i].flatten().cpu().numpy().astype(np.int16).tolist())
                all_gmm_list.append((params_HL_list[c][i]*10).permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0).cpu().numpy().astype(np.int).tolist())
                
            for c in range(3):
                all_subband_list.append(LH_list[c][i].flatten().cpu().numpy().astype(np.int16).tolist())
                all_gmm_list.append((params_LH_list[c][i]*10).permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0).cpu().numpy().astype(np.int).tolist())
                
            for c in range(3):
                all_subband_list.append(HH_list[c][i].flatten().cpu().numpy().astype(np.int16).tolist())
                all_gmm_list.append((params_HH_list[c][i]*10).permute(1, 0, 2, 3).flatten(start_dim=1).permute(1, 0).cpu().numpy().astype(np.int).tolist())
                

            
        print("===============Get the entropy params of the subbands succeed!=================")
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())
        ans_bin = coder.coding(all_gmm_list, all_subband_list , all_scale)
        print(f'BPP: {len(ans_bin)*8/(3840*2160)}')
        ans_bin = bytearray(ans_bin)
        
        write_bin(ans_bin,os.path.dirname(bin_path) + "/v3_fix_gmm.bin")
        print('(height + pad_h): ', (height + pad_h))
        print('(width + pad_w)', (width + pad_w))
        print('(height + pad_h) * (width + pad_w) * 2:', (height + pad_h) * (width + pad_w) * 2)
        print("===============Arithmetic coding succeed!=================")
        # inverse_transform
        recon_lst = []
        # patch2subband
        # LL_list, HL_list, LH_list, HH_list = patch2subbands(LL, HL_list, LH_list, HH_list, patch_num, y_subband_h, y_subband_w, uv_subband_h, uv_subband_w, code_block_size)
        for c in range(3):
            LL = LL_list[c] * trans_used_scale
            for i in range(trans_steps):
                j = trans_steps - 1 - i
                HL = HL_list[c][i] * trans_used_scale
                LH = LH_list[c][i] * trans_used_scale   
                HH = HH_list[c][i] * trans_used_scale    
                LL = models_dict['transform'].inverse_trans(LL, HL, LH, HH, j, ishalf=True)
            recon_lst.append(LL.to(torch.float32))
        yuv444_u = F.interpolate(recon_lst[1], scale_factor=2, mode='bicubic', align_corners=True).to('cuda')
        yuv444_v = F.interpolate(recon_lst[2], scale_factor=2, mode='bicubic', align_corners=True).to('cuda')

        h, w = height, width
        yuv444_y = recon_lst[0][:, :, :h, :w]
        yuv444_u = yuv444_u[:, :, :h, :w]
        yuv444_v = yuv444_v[:, :, :h, :w]
        recon = torch.cat((yuv444_y, yuv444_u, yuv444_v), dim=0)
        print(recon.shape)

        recon_Y = recon_lst[0].permute(1, 0, 2, 3)
        recon_Y = recon_Y[:, :, 0:height, 0:width]

        recon = yuv2rgb(recon.permute(1, 0, 2, 3))
        recon = recon[:, :, 0:height, 0:width]

        recon = torch.clamp(torch.round(recon), 0., 255.)
        
        mse = torch.mean((recon.cpu() - original_img) ** 2)
        psnr = (10. * torch.log10(255. * 255. / mse)).item()
        print(f'PSNR before post: {psnr}')
        recon_save = recon[0, :, :, :]
        recon_save = recon_save.permute(1, 2, 0)
        recon_save = recon_save.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon_save, 'RGB')
        img.save(os.path.join(args.recon_dir, bin_name + 'before_post.png'))

        if args.isPostGAN:

            recon = models_dict['post'](recon)

            if height * width > 1080 * 1920:
                h_list = [0, height // 2, height]
                w_list = [0, width // 2, width]
                k_ = 2
            else:
                h_list = [0, height]
                w_list = [0, width]
                k_ = 1
            gan_rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width) - w_list[_j + 1]
                    tmp = models_dict['postGAN'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                                 w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    gan_rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                               -pad_start_h:tmp.size()[
                                                                                                                2] - pad_end_h,
                                                                                               -pad_start_w:tmp.size()[
                                                                                                                3] - pad_end_w]
            recon = gan_rgb_post
        else:

            h_list = [0, height // 3, height // 3 * 2, height]
            w_list = [0, width // 3, width // 3 * 2, width]
            k_ = 3
            rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width) - w_list[_j + 1]
                    tmp = models_dict['post'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                              w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                           -pad_start_h:tmp.size()[
                                                                                                            2] - pad_end_h,
                                                                                           -pad_start_w:tmp.size()[
                                                                                                            3] - pad_end_w]
            recon = rgb_post

        recon = torch.clamp(torch.round(recon), 0., 255.)
        print(f'recon.shape:{recon.shape}')


        test_yuv = rgb2yuv(recon)

        test_y = torch.unsqueeze(test_yuv[0][0], dim=0)
        test_y = torch.unsqueeze(test_y, dim=0)

        mse = torch.mean((recon.cpu() - original_img) ** 2)
        print(f'test_y.shape:{test_y.shape}')
        print(f'ori_y.shape:{original_y.shape}')

        psnr = (10. * torch.log10(255. * 255. / mse)).item()

        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(os.path.join(args.recon_dir, bin_name + '.png'))

        print('encoding finished!')
        logfile.write('encoding finished!' + '\n')
        end = time.time()
        print('Encoding-time: ', end - start)
        logfile.write('Encoding-time: ' + str(end - start) + '\n')

        print('bit_out closed!')
        logfile.write('bit_out closed!' + '\n')

        print('PSNR_RGB: ', psnr)
        logfile.write('PSNR_RGB: ' + str(psnr) + '\n')
        logfile.flush()
    logfile.close()
