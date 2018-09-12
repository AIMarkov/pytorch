#pytorch中多进程程的不同进程网络梯度共享实现
import torch
from torch.autograd import Variable
import torch.nn as nn
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear=nn.Linear(2,1)
    def forward(self, input):
        return self.linear(input)
test=nn.Linear(2,1)


x=Variable(torch.ones(1,2))
print('x:',x)

#在网络更新过程中,也就是loss function backward()后,网络的parameters就是一个包含梯度迭代器
    #Net=NN()
    # print(Net.linear.weight.data)
    # opti=torch.optim.SGD(Net.parameters(),lr=1e-3)
    # loss=Net(x)-6
    # opti.zero_grad()
    # loss.backward()
    # print('grad:',[i._grad for i in Net.parameters()])
    # #print('parameters:',torch.nn.utils.parameters_to_vector(Net.parameters()))
    # opti.step()
    # print(Net.linear.weight.data)
#用一个网络的梯度更新另外一个网络
Net0=NN()
Net1=NN()
Net1.load_state_dict(Net0.state_dict())
# for p,q in zip(Net0.parameters(),Net1.parameters()):
#     p.data.copy_(q.data)
print('oldNet1:',Net1.linear.weight.data)
print('oldNet0:',Net0.linear.weight.data)
Optim0=torch.optim.SGD(Net0.parameters(),lr=1)
Optim1=torch.optim.SGD(Net1.parameters(),lr=1)
loss=Net0(x)-6

Optim0.zero_grad()
Optim1.zero_grad()
loss.backward()
nn.utils.clip_grad_norm(Net0.parameters(),max_norm=0.1)#裁剪梯度,最大值为5
for q,p in zip(Net1.parameters(),Net0.parameters()):
    q.grad=p.grad
    print('grad:',p.grad)
Optim0.step()
Optim1.step()
print('newNet0:',Net0.linear.weight.data)
print('newNet1:',Net1.linear.weight.data)

