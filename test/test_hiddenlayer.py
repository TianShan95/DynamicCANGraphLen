import torch
import torchvision.models
import hiddenlayer as hl

# VGG16 with BatchNorm
model = torchvision.models.vgg16()

# Build HiddenLayer graph
# Jupyter Notebook renders it automatically
graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
graph.save(f"actor.png", format='png')