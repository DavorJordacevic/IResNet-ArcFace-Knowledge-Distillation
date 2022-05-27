
import cv2
import time
import torch
import argparse
import numpy as np

from backbones import get_model

@torch.no_grad()
def inference(network):
    img = cv2.imread("image.jpg")
    #img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = torch.jit.load(network)

    #start = time.time()
    encoding = net(img).numpy()
    #end = time.time()
    print(encoding)
    #print(f'Time: {round((end-start)*1000, 2)}ms.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace')
    parser.add_argument('--network', type=str, default='ArcfaceR50.ptl', help='backbone network')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.network)
