import torch
x = torch.rand(5, 3)
print(x)
print('Using {} device version {}'.format(torch.cuda.is_available(), torch.version.cuda))