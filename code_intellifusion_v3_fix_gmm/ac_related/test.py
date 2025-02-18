import struct
import numpy as np
import arithmetic_coding as coder
from tool import *
def load_gmms_from_bin(filename):
    with open(filename, 'rb') as f:
        size_data = f.read(8)  # 根据平台确定 size_t 的大小，通常是 8 字节
        size = struct.unpack('Q', size_data)[0]  # 'Q' 表示 unsigned long long (8 字节)
        num_ints = size * 9
        gmm_data = f.read(num_ints * 4)
        gmms = struct.unpack(f'{num_ints}i', gmm_data)
        gmms = np.array(gmms, dtype=np.int32).reshape(size, 9)
    return gmms, size

def load_array_from_bin(filename):
    with open(filename, 'rb') as f:
        size_data = f.read(8)  # 根据平台确定 size_t 的大小，通常是 8 字节
        size = struct.unpack('Q', size_data)[0]  # 'Q' 表示 unsigned long long (8 字节)
        array = np.frombuffer(f.read(size * 2), dtype=np.int16)
    return array, size


all_data = []
all_gmm = []
all_scale = []
for i in range(39):
    data_path = f'bins/data_sacle_10_{i}.bin'
    gmm_path = f'bins/gmm_sacle_10_{i}.bin'
    data, data_size = load_array_from_bin(data_path)
    gmm, gmm_size = load_gmms_from_bin(gmm_path)
    all_data.append(data)
    all_gmm.append(gmm)
    all_scale.append(10)
    
all_gmm_list = [gmm.tolist() for gmm in all_gmm]  
all_data_list = [data.tolist() for data in all_data]  
# 熵编码
ans_bin = coder.coding(all_gmm_list, all_data_list , all_scale)
ans_bin = bytearray(ans_bin)

write_bin('bins/ans.bin', ans_bin)

infos = dec_split('bins/ans.bin', all_scale)
success_cnt = 0

for idx, info in enumerate(infos):
    blk_bin = info['bin']
    write_bin(blk_bin, f'enc_bins/blk_{idx}.bin')
    xmin, xmax, gmm_scale = info['xmin'], info['xmax'], info['gmm_scale']
    blk_len = info['blk_len']
    
    dec_res = []
    # 基于块，初始化一个类
    dec = coder.ArithmeticPixelDecoder(32, f'enc_bins/blk_{idx}.bin', blk_len,gmm_scale, xmin, xmax)
    # 获取这个块的gmm
    blk_gmm = get_blk_gmm(all_gmm, idx).tolist()
    # 一个个解码
    for i in range(0, blk_len):
        # 获取该元素的gmm
        p_gmm = blk_gmm[i]
        x = dec.read(p_gmm)
        dec_res.append(x)
    
    ori_data = get_blk_data(all_data, idx).tolist()
    cmp_res = [x == y for x, y in zip(dec_res, ori_data)]
    if sum(cmp_res) == len(cmp_res):
        success_cnt += 1

print(f'success rate: {success_cnt}/{len(infos)}')
    