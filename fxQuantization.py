
import io
import cv2
import time
import copy
import torch
import argparse
import numpy as np
from backbones import get_model
import torch.quantization.quantize_fx as quantize_fx
from torch.utils.mobile_optimizer import optimize_for_mobile


@torch.no_grad()
def convert(weight, name):
    x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).float()
    x.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight, map_location={'cuda:0': 'cpu'}))
    model_to_quantize = copy.deepcopy(net)
    model_to_quantize.eval()

    #
    # post training static quantization
    #
    qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
    model_to_quantize.eval()
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
    # calibrate (not shown)
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    torch.save(model_quantized.state_dict(), "ArcfaceFXQuant" + name.upper() + ".pth")

    traced_script_module = torch.jit.trace(model_quantized, x)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("ArcfaceFXQuant" + name.upper() + ".ptl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()
    convert(args.weight, args.network)