import torch
import torch.nn as nn
import math

model_urls = {
    'resnet18':'../resnet_pth/resnet18-5c106cde.pth',
    'resnet34':'../resnet_pth/resnet34-333f7ec4.pth',
    'resnet50':'../resnet_pth/resnet50-19c8e357.pth',
    'resnet101':'../resnet_pth/resnet101-5d3b4d8f.pth',
    'resnet152':'../resnet_pth/resnet152-b121ed2d.pth'
}

class Bottleneck(nn.Module):
    expansion = 4