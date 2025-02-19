import torch.nn as nn

import torch
q = torch.randn(1,1,162,3696)


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.convid = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,padding=1)
        # self.pad=nn.ReflectionPad2d((0,0,0,1))
        # self.Mp1=nn.MaxPool2d(2)
        # self.convid2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,padding=1)
        # self.Mp2=nn.MaxPool2d(2)
        # self.flat=nn.Flatten()
        # self.linner1 = nn.Linear(136000,10000)
        # self.linner2 = nn.Linear(10000,100)
        # self.linner3 = nn.Linear(100,1)
        #Squential队列
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,padding=1),
            #nn.ReflectionPad2d((0,0,0,1)),
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,padding=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(110880,10000),
            nn.Linear(10000,100),
            nn.Linear(100,1)
        )

    def forward(self,x):
        x=self.model(x)
        return x
    
tes = Model()
print(tes)
input = q
out= tes(input)
print(out.squeeze())