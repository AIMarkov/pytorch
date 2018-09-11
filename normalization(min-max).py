import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as pl
class LayerNorm(nn.Module):  #做一个归一化处理,使特征在(0,1)之间
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) #nn.Parameter也是一种Variable#1
        self.beta = nn.Parameter(torch.zeros(features))  #0
        self.eps = eps#这里self.eps是为了避免分母为0设置的

    def forward(self, x):
        min = x.min(-1, keepdim=True)#均值,后面参数表示保持维度数量不变,因为对某一维度求均值后，维度数量会减1，TRUR表示保持维度不变,这样才能相减。
        max = x.max(-1, keepdim=True)#标准差
        print('x:',x)
        print('min:',min[0])#min,max函数输出两个，一个最大，最小值，另一个稀疏矩阵
        print('max:',max[0])


        result=self.gamma * (x - min[0]) / ((max[0]-min[0]) + self.eps)

        return result
nomal=LayerNorm(3)#特征维度是2
x=np.random.randn(5,3)*100#批量是3
x=Variable(torch.Tensor(x))
result=nomal(x)
print(result)

