
import cv2
import time
import copy
import torch
import argparse
import numpy as np
from backbones import get_model
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.jit.mobile import _backport_for_mobile,_get_model_bytecode_version


@torch.no_grad()
def convert(weight, name):
    x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).float()
    x.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight, map_location={'cuda:0': 'cpu'}))
    net.eval()
    
    file_name = "Arcface" + name.upper() + ".ptl"
    traced_script_module = torch.jit.trace(net, x)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(file_name)
    #_backport_for_mobile(f_input=file_name, f_output=file_name, to_version=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()
    convert(args.weight, args.network)