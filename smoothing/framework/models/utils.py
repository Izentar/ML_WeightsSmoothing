import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)