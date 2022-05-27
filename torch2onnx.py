
import cv2
import time
import torch
import argparse
import numpy as np

from backbones import get_model

@torch.no_grad()
def convert(weight, name):
    x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).float()
    x.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location={'cuda:0': 'cpu'}))
    net.eval()

    # Export the model
    torch.onnx.export(net,               # model being run
        x,                         # model input (or a tuple for multiple inputs)
        "Arcface" + name.upper() + ".onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()
    convert(args.weight, args.network)