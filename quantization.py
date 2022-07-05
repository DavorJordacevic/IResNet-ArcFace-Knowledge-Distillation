
import cv2
import time
import copy
import torch
import argparse
import numpy as np
from backbones import get_model
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization.observer import HistogramObserver, MovingAveragePerChannelMinMaxObserver

@torch.no_grad()
def convert(weight, name):
    x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).float()
    x.div_(255).sub_(0.5).div_(0.5)
    x = x.to('cpu')
    quant = torch.quantization.QuantStub()
    x_quant = quant(x)

    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location={'cuda:0': 'cpu'}))
    net.eval()
    
    model_to_quantize = copy.deepcopy(net)
    model_to_quantize.eval()
    
    # set the qconfig for PTQ
    model_to_quantize.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # set the qengine to control weight packing
    torch.backends.quantized.engine = 'qnnpack'
    model_to_quantize = torch.quantization.prepare(model_to_quantize, inplace=False)
    with torch.no_grad():
        model_to_quantize(x_quant)
    model_to_quantize = torch.quantization.convert(model_to_quantize, inplace=False)

    torch.save(model_to_quantize.state_dict(), "ArcfaceQuant" + name.upper() + ".pth")
    
    """
    with torch.no_grad():
        device = 'cpu'
        x_quant.to(device)
        model_to_quantize.to(device)
        traced_script_module = torch.jit.trace(model_to_quantize, x_quant)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized._save_for_lite_interpreter("ArcfaceQuant" + name.upper() + ".ptl")
    
    torch.onnx.export(model_to_quantize,           # model being run
        x,                                         # model input (or a tuple for multiple inputs)
        "ArcfaceQuant" + name.upper() + ".onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,                        # store the trained parameter weights inside the model file
        opset_version=10,                          # the ONNX version to export the model to
        do_constant_folding=True)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='ArcFaceR50.ptl', help='backbone network')
    parser.add_argument('--weight', type=str, default='')

    args = parser.parse_args()
    convert(args.weight, args.network)