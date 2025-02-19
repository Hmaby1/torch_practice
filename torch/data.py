from dscribe.descriptors import SOAP
from ase.build import molecule
from ase.atoms import Atoms
from ase import io
from pymatgen.core import Structure
from module import Model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
species = ["Y", "Sb", "Te"]
r_cut = 3.5
n_max = 8
l_max = 6

class soap_data():
    def __init__(self,
                 structure,
                 species,
                 periodic:bool=True,
                 r_cut:float=3.5,
                 n_max:float=8.0,
                 l_max:float=6.0,):
        self.stucture=structure
        self.ase_stru = structure.to_ase_atoms()
        self.species = list(set(self.ase_stru.get_chemical_symbols()))
        self.soap= SOAP(
            species=self.species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
        )
        self.soap_descriptor = self.soap.create(self.ase_stru)
        self


# 加载结构和描述符
path=r'D:\Desk\work_file\VASP\YST-1\6\6.vasp'
yst_structure=io.read(path)
soap= SOAP(
    species=list(set(yst_structure.get_chemical_symbols())),
    periodic=True,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
)
descriptor = soap.create(yst_structure)
print(descriptor.shape)

#tensor data
tensor_data = torch.tensor(descriptor, dtype=torch.float32)
tensor_data=tensor_data.reshape(1,1,162,3696)

energy = -520.396
energy_data = torch.tensor(energy, dtype=torch.float32).unsqueeze(0)

dataset = TensorDataset(tensor_data,energy_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#加载model
model=Model()

# 训练模型
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 300  # 训练 1000 轮
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        output = model(X_batch)
        output = output.squeeze()
        optimizer.zero_grad()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Energy_cal={output},target=-520.396')
    print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')
