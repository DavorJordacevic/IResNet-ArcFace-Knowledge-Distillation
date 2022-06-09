#!/bin/sh
cd ..
python torch2onnx.py --network=r18 --weight=weights/r18/backbone.pth
python torch2onnx.py --network=r34 --weight=weights/r34/backbone.pth
python torch2onnx.py --network=r50 --weight=weights/r50/backbone.pth
python torch2onnx.py --network=r100 --weight=weights/r100/backbone.pth