import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


model = models.alexnet()
print(model)