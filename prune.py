import cv2
import time
import torch
import argparse
import numpy as np
import torch_pruning as tp

from backbones import get_model

parser = argparse.ArgumentParser(description='PyTorch ArcFace')
parser.add_argument('--network', type=str, default='r50', help='backbone network')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--img', type=str, default=None)
args = parser.parse_args()

# 0. Load the model
model = get_model(args.network, fp16=True)
model.load_state_dict(torch.load(args.weight))
model.eval()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,112,112))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.conv1.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9, ...]
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
print(pruning_plan)

# 4. execute this plan (prune the model)
pruning_plan.exec()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L2Strategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,112,112))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.conv1.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9, ...]
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
print(pruning_plan)

# 4. execute this plan (prune the model)
pruning_plan.exec()

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.RandomStrategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,112,112))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.conv1.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9, ...]
pruning_plan = DG.get_pruning_plan( model.conv1, tp.prune_conv, idxs=pruning_idxs )
print(pruning_plan)

# 4. execute this plan (prune the model)
pruning_plan.exec()
# 5. save a pruned model
# torch.save(model.state_dict(), 'model.pth') # weights only
torch.save(model, 'arcfacer50pruned.pth') # obj (arch + weights), recommended.

# Export the model to onnx
x = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = np.transpose(x, (2, 0, 1))
x = torch.from_numpy(x).unsqueeze(0).float()
x.div_(255).sub_(0.5).div_(0.5)

torch.onnx.export(model,               # model being run
                x,                         # model input (or a tuple for multiple inputs)
                "arcfacer50pruned.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True)
