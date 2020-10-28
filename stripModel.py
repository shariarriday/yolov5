import argparse
import glob
import json
import os
import shutil
from pathlib import Path
import pandas as pd

import numpy as np
import torch
import yaml
from tqdm import tqdm

# from models.yolo import Model
# from models.experimental import attempt_load
from utils.general import strip_optimizer
from utils.torch_utils import select_device


def strip_model(weights=None,
                save = '',
                device='cpu'):
    try:
        model = strip_optimizer(weights, s=save)
    except:
        print('Please ensure you have given correct path to weights. Otherwise contact developer')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
    parser.add_argument('--save-path', type=str, default='', help='saving model as')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = select_device(opt.device)
    print(opt)

    strip_model(
            opt.weights,
            opt.save_path,
            device)

