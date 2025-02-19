import torch
from module import Model
m = Model()
optim = torch.optim.Adam(m.parameters,lr=1e-4)