import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
class net1(nn.Module):
    def __init__(self):
        super(net1,self).__init__()
        self.layer=nn.Linear(2,3)
    def forward(self, input):
        return self.layer(input)
class net2(nn.Module):
    def __init__(self):
        super(net2,self).__init__()
        self.layer1=net1()
    def forward(self, input):
        return self.layer1(input)
x=Variable(torch.ones([1,2]))
y=Variable(torch.ones([1,3]))
model=net2()
modeloptimizer=optim.Adam(model.parameters(),lr=1e-2)
for i in range(1000):
    weight=model.layer1.layer.weight.data
    print(weight)
    Y=model(x)
    loss=F.smooth_l1_loss(Y, y.detach())
    modeloptimizer.zero_grad()
    loss.backward()
    modeloptimizer.step()
print(Y)