import torch.nn as nn
import torch
import numpy as np


GRU=nn.GRU(2,2,num_layers=1,dropout=0,bidirectional=False,batch_first=True)
input=torch.Tensor(np.random.normal(0,0.1,[1,2,2]))
print("input:",input)
hidde_init=torch.Tensor(np.random.normal(0,0.1,[1,1,2]))#为甚hidden_init是1,2,2,一个是hidden_size,一个是批量数
ht,ct=GRU(input,hidde_init)##GRU没有Ct输出因此第二个输出表示最后一个隐层状态
print("ht:",ht)
print("ct:",ct)

