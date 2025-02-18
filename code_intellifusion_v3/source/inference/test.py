import torch
import os
import Model
from utils import subband2patch, patch2subband
from PixelCNN_light import PixelCNN
path1 = "/data/kangbw/models/hardware_project/output/enc_coefficient/v3/"
path2 = "/data/kangbw/models/hardware_project/output/decode/v3/"
path3 = "/data/kangbw/models/hardware_project/output/enc_params/"

LL_enc = torch.load(os.path.join(path1, f'LL.pt'))
# HL_list_03_dec = torch.zeros_like(HL_list_03_enc)
LL_dec = torch.load(os.path.join(path2, f'LL.pt'))
# print(torch.equal(HL_list_03_enc, HL_list_03_dec))
difference = LL_dec != LL_enc
indices = torch.nonzero(difference, as_tuple=True)
print(indices)
print(LL_enc[indices[0], indices[1], indices[2], indices[3]])
print(LL_dec[indices[0], indices[1], indices[2], indices[3]])
# print(LL_enc[:, :, 0:10, 0:10])
# print(LL_dec[:, :, 0:10, 0:10])
# HL_03_params = torch.load(os.path.join(path3, f"params_HL_list_03.pt"))
# LL_params = torch.load(os.path.join(path3, f"params_LL.pt"))
# print(HL_03_params[:, :, 0, 0])
# torch.set_printoptions(profile="full")
# print(LL_enc[:, :, 0, 0])
import pdb; pdb.set_trace()
# print(HL_list_03_enc[:, :, -10:])
# print(HL_list_03_dec[:, :, -10:])
# print(HL_03_params[:, :, 0, :6])
# print(LL_0_params[:, :, 0, :7])
# x = torch.arange(25).reshape(1, 1, 5, 5).to(torch.float32)
# model = PixelCNN()
# y = model(x, decode=True)
# print(x)
# print(y)

# x[:, :, 3:, :] = 0
# x[:, :, 2, 3:] = 0
# y = model(x, decode=True)
# print(x)
# print(y)
# model_path = '/data/kangbw/models/hardware_project/model/half/'
# init_scale = 8.0
# checkpoint = torch.load(model_path + '/' + str(0.4) + '_mse.pth')
# all_part_dict = checkpoint['state_dict']

# models_dict = {}
# models_dict['transform'] = Model.Transform_aiWave(init_scale, isAffine=False)
# models_dict['entropy'] = Model.Entropy_coding(init_scale, isAffine=False)
# models_dict['post'] = Model.Post()
# models_dict_update = {}
# for key, model in models_dict.items():
#     myparams_dict = model.state_dict()
#     new_dict = {}
#     # change the name of state_dict when load the entropy model
#     if 'entropy' in key:
#         for k, v in list(all_part_dict.items()):
#             if 'coding' in k:
#                 prefix = k.split('.')[0] 
#                 k = prefix + '.' + k
#                 new_dict[k] = v
#             elif 'wavelet_transform' or 'scale_net' in k:
#                 k = 'transform.' + k
#                 new_dict[k] = v
#             else:
#                 new_dict[k] = v          
#     else:
#         new_dict = all_part_dict     
#     part_dict = {k: v for k, v in new_dict.items() if k in myparams_dict}
#     myparams_dict.update(part_dict)
#     model.load_state_dict(myparams_dict)
#     if torch.cuda.is_available():
#         model = model.cuda()
#         # transform with fp16
#         if key == 'transform':
#             model = model.half()
#         model.eval()
#     models_dict_update[key] = model
# models_dict.update(models_dict_update)
# used_scale = models_dict['entropy'].transform.used_scale(ishalf=False)
# params_dec = models_dict['entropy'].coding_HL_list(torch.cat((HL_list_03_dec, LL_dec*used_scale), 1), 3)[:, :, 0, 0]
# print(params_dec)
# x = torch.load(os.path.join(path1, f'LL_list0'))
# x_ref = torch.zeros(1, 1, 5, 5).cuda()
# params_enc = models_dict['entropy'].coding_LL(x)
# print(params_enc[:, :, 0, :6])
# x_ref[:, :, 2, 0] = x[:, :, 0, 1]
# x_ref[:, :, 2, 1] = x[:, :, 0, 2]
# print('=========================')
# print(x_ref)
# params_dec = models_dict['entropy'].coding_LL(x_ref, decode=True)
# print(params_dec)
# diff = [i for i in range(len(list1)) if list1[i] != list2[i]]
# print(diff)

x = torch.arange(64).reshape(1, 1, 8, 8)
print(x)
y = img2patch(x, 4)
print(y)
x_recon = patch2img(y, 4, 8, 8)
print(x_recon)