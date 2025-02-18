import argparse
import json

import torch.cuda
from enc_lossless import enc_lossless
from encode_lossy import enc_lossy

parser = argparse.ArgumentParser(description='Hardware project', conflict_handler='resolve')
parser.add_argument('--cfg_file', type=str, default=r'../../cfg/encode_yuv.cfg')

args, unknown = parser.parse_known_args()


cfg_file = args.cfg_file

with open(cfg_file, 'r') as f:
    cfg_dict = json.load(f)

    for key, value in cfg_dict.items():
        if isinstance(value, int):
            parser.add_argument('--{}'.format(key), type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument('--{}'.format(key), type=float, default=value)
        else:
            parser.add_argument('--{}'.format(key), type=str, default=value)

cfg_args, unknown = parser.parse_known_args()


def main():
    args = parser.parse_args()

    print(args)

    if args.isLossless == 0:
        enc_lossy(args)
    else:
        enc_lossless(args)

if __name__ == "__main__":
    main()
