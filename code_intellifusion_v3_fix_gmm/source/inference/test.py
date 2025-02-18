import torch
import os
import Model
from utils import subband2patch, patch2subband, patch2subbands
from PixelCNN_light import PixelCNN

path1 = "/data/kangbw/models/hardware_project/output/enc_coefficient/v3_fix_gmm/"
path2 = "/data/kangbw/models/hardware_project/output/decode/v3_fix_gmm/"

path3 = "/data/kangbw/models/hardware_project/output/enc_params/v3_fix_gmm/"
path4 = "/data/kangbw/models/hardware_project/output/decode/dec_params/v3_fix_gmm/"

LL_enc = torch.load(path1 + "LL.pt")
LL_dec = torch.load(path2 + "LL.pt")
HL_enc_list = []
LH_enc_list = []
HH_enc_list = []
HL_dec_list = []
LH_dec_list = []
HH_dec_list = []

for i in range(2):
    j = 4 - 1 - i
    HL_enc_list.append(torch.load(path1 + "HL_list_" + str(j) + ".pt"))
    LH_enc_list.append(torch.load(path1 + "LH_list_" + str(j) + ".pt"))
    HH_enc_list.append(torch.load(path1 + "HH_list_" + str(j) + ".pt"))
    HL_dec_list.append(torch.load(path2 + "HL_list_" + str(j) + ".pt"))
    LH_dec_list.append(torch.load(path2 + "LH_list_" + str(j) + ".pt"))
    HH_dec_list.append(torch.load(path2 + "HH_list_" + str(j) + ".pt"))

LL_dif = LL_dec != LL_enc
indices = torch.nonzero(LL_dif, as_tuple=True)
print("LL_dif_num: ", len(indices[0]))
print("dif: LL_enc: ", LL_enc[indices[0], indices[1], indices[2], indices[3]])
print("dif: LL_dec: ", LL_dec[indices[0], indices[1], indices[2], indices[3]])

for i in range(2):
    j = 4 - 1 - i
    HL_dif = HL_dec_list[i] != HL_enc_list[i]
    indices = torch.nonzero(HL_dif, as_tuple=True)
    print(f"HL_list_{j}_dif_num: ", len(indices[0]))
    print("indices: ", indices[0], indices[1], indices[2], indices[3])
    print("dif: HL_enc: ", HL_enc_list[i][indices[0], indices[1], indices[2], indices[3]])
    print("dif: HL_dec: ", HL_dec_list[i][indices[0], indices[1], indices[2], indices[3]])
    
    LH_dif = LH_dec_list[i] != LH_enc_list[i]
    indices = torch.nonzero(LH_dif, as_tuple=True)
    print(f"LH_list_{j}_dif_num: ", len(indices[0]))
    print("indices: ", indices[0], indices[1], indices[2], indices[3])
    print("dif: LH_enc: ", LH_enc_list[i][indices[0], indices[1], indices[2], indices[3]])
    print("dif: LH_dec: ", LH_dec_list[i][indices[0], indices[1], indices[2], indices[3]])
    
    HH_dif = HH_dec_list[i] != HH_enc_list[i]
    indices = torch.nonzero(HH_dif, as_tuple=True)
    print(f"HH_list_{j}_dif_num: ", len(indices[0]))
    print("indices: ", indices[0], indices[1], indices[2], indices[3])
    print("dif: HH_enc: ", HH_enc_list[i][indices[0], indices[1], indices[2], indices[3]])
    print("dif: HH_dec: ", HH_dec_list[i][indices[0], indices[1], indices[2], indices[3]])

# LL_params_enc = torch.load(path3 + "params_LL.pt").cuda()
# LL_params_dec = torch.load(path4 + "params_LL.pt").cuda()

# HL_params_enc_list = []
# LH_params_enc_list = []
# HH_params_enc_list = []

# HL_params_dec_list = []
# LH_params_dec_list = []
# HH_params_dec_list = []

# for i in range(4):
#     j = 4 - 1 - i
#     HL_params_enc_list.append(torch.load(path3 + "params_HL_list_" + str(j) + ".pt").cuda())
#     LH_params_enc_list.append(torch.load(path3 + "params_LH_list_" + str(j) + ".pt").cuda())
#     HH_params_enc_list.append(torch.load(path3 + "params_HH_list_" + str(j) + ".pt").cuda())
    
#     HL_params_dec_list.append(torch.load(path4 + "params_HL_list_" + str(j) + ".pt").cuda())
#     LH_params_dec_list.append(torch.load(path4 + "params_LH_list_" + str(j) + ".pt").cuda())
#     HH_params_dec_list.append(torch.load(path4 + "params_HH_list_" + str(j) + ".pt").cuda())

# # Compare the enc and dec params
# LL_diff = torch.sum(torch.abs(LL_params_enc - LL_params_dec)).item()
# HL_diff_list = []
# LH_diff_list = []
# HH_diff_list = []
# for i in range(4):
#     j = 4 - 1 - i
#     HL_diff_list.append(torch.sum(torch.abs(HL_params_enc_list[j] - HL_params_dec_list[j])).item())
#     LH_diff_list.append(torch.sum(torch.abs(LH_params_enc_list[j] - LH_params_dec_list[j])).item())
#     HH_diff_list.append(torch.sum(torch.abs(HH_params_enc_list[j] - HH_params_dec_list[j])).item())
    
# print("LL_diff: ", LL_diff)
# print("HL_diff_list: ", HL_diff_list)
# print("LH_diff_list: ", LH_diff_list)
# print("HH_diff_list: ", HH_diff_list)
    