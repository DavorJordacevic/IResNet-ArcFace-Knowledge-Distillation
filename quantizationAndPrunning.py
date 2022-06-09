
import cv2
import time
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from backbones import get_model
from torch.nn.utils import prune
from torch.utils.mobile_optimizer import optimize_for_mobile


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model

def prune_model_ln_structured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    return model

def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model

@torch.no_grad()
def convert(weight, name, layer_type, proportion):
    x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).float()
    x.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location={'cuda:0': 'cpu'}))
    model_to_optimize = copy.deepcopy(net)
    model_to_optimize.eval()
    
    # set the qconfig for PTQ
    qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # set the qengine to control weight packing
    torch.backends.quantized.engine = 'qnnpack'
    torch.quantization.prepare(model_to_optimize)
    torch.quantization.convert(model_to_optimize)
    
    model_to_prune = prune_model_l1_unstructured(model_to_optimize, layer_type, proportion)
    #model_to_prune = prune_model_ln_structured(model_to_prune, layer_type, proportion)
    #model_to_prune = prune_model_global_unstructured(model_to_prune, layer_type, proportion)

    torch.save(model_to_prune.state_dict(), "ArcfaceQP" + name.upper() + ".pth")

    traced_script_module = torch.jit.trace(model_to_optimize, x)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter("ArcfaceQP" + name.upper() + ".ptl")

    torch.onnx.export(model_to_optimize,         # model being run
        x,                                       # model input (or a tuple for multiple inputs)
        "ArcfaceQP" + name.upper() + ".onnx",    # where to save the model (can be a file or file-like object)
        export_params=True,                      # store the trained parameter weights inside the model file
        opset_version=10,                        # the ONNX version to export the model to
        do_constant_folding=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()
    convert(args.weight, args.network, nn.Conv2d, 0.5)