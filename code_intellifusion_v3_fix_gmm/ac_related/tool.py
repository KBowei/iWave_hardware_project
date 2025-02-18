import numpy as np
import os


def write_bin(data, file):
    try:
        # 获取文件夹路径
        folder = os.path.dirname(file)
        
        # 如果文件夹不存在，则创建文件夹
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 打开文件并写入数据
        with open(file, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False


def dec_split(binfile, gmm_scales):
    try:
        with open(binfile, 'rb') as f:
            buffer = f.read()
    except Exception as e:
        print(f"Error opening file: {e}")
        return []

    file_size = len(buffer)
    if file_size <= 0:
        print("File size is invalid.")
        return []

    offset = 0
    xmins = np.empty(39, dtype=np.int16)
    xmaxs = np.empty(39, dtype=np.int16)

    for i in range(39):
        # 读取 xmins[i]
        xmins[i] = np.frombuffer(buffer[offset:offset + 2], dtype=np.int16)[0]
        offset += 2

        # 读取 xmaxs[i]
        xmaxs[i] = np.frombuffer(buffer[offset:offset + 2], dtype=np.int16)[0]
        offset += 2

    ratios = [16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    blk_size = 64
    blk_count = 3257
    bins = buffer[offset + blk_count * 2:]
    infos = []

    for sub in range(13):
        for ch in range(3):
            H = 2160 if ch == 0 else 1088
            W = 3840 if ch == 0 else 1920
            h = H // ratios[sub]
            w = W // ratios[sub]
            idx = sub * 3 + ch

            for row in range(0, h, blk_size):
                for col in range(0, w, blk_size):
                    row_end = min(row + blk_size, h)
                    col_end = min(col + blk_size, w)
                    blk_len = (row_end - row) * (col_end - col)

                    bin_len = int.from_bytes(buffer[offset:offset + 2], byteorder='little')
                    offset += 2

                    bin_data = bins[:bin_len]
                    bins = bins[bin_len:]

                    info = {
                        "xmin": xmins[idx],
                        "xmax": xmaxs[idx],
                        "gmm_scale": gmm_scales[idx],
                        "blk_len": blk_len,
                        "bin_len": bin_len,
                        "bin": bin_data
                    }
                    infos.append(info)

    return infos


def get_blk_gmm(gmms, blk_idx):
    ratios = [16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    blk_size = 64

    blk_p = 0
    for sub in range(13):
        for ch in range(3):
            H = 2160 if ch == 0 else 1088
            W = 3840 if ch == 0 else 1920
            h = H // ratios[sub]
            w = W // ratios[sub]
            idx = sub * 3 + ch
            gmm = gmms[idx]

            for row in range(0, h, blk_size):
                for col in range(0, w, blk_size):
                    if blk_p != blk_idx:
                        blk_p += 1
                        continue

                    row_end = min(row + blk_size, h)
                    col_end = min(col + blk_size, w)
                    blk_len = (row_end - row) * (col_end - col)

                    blk_gmm = np.zeros((blk_len,9), dtype=np.int)
                    for i in range(row, row_end):
                        for j in range(col, col_end):
                            blk_gmm[(i - row) * (col_end - col) + (j - col)] = np.int16(gmm[i * w + j])

                    return blk_gmm
    return None


def get_blk_data(datas, blk_idx):
    ratios = [16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    blk_size = 64

    blk_p = 0
    for sub in range(13):
        for ch in range(3):
            H = 2160 if ch == 0 else 1088
            W = 3840 if ch == 0 else 1920
            h = H // ratios[sub]
            w = W // ratios[sub]
            idx = sub * 3 + ch
            sub_data = datas[idx]

            for row in range(0, h, blk_size):
                for col in range(0, w, blk_size):
                    if blk_p != blk_idx:
                        blk_p += 1
                        continue

                    row_end = min(row + blk_size, h)
                    col_end = min(col + blk_size, w)
                    blk_len = (row_end - row) * (col_end - col)

                    blk_data = np.zeros((blk_len,), dtype=np.int16)
                    for i in range(row, row_end):
                        for j in range(col, col_end):
                            blk_data[(i - row) * (col_end - col) + (j - col)] = sub_data[i * w + j]

                    return blk_data
    return None
