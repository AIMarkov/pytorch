import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as pl
class LayerNorm(nn.Module):  #做一个标准化处理,使特征均值为0，标准差为1
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) #nn.Parameter也是一种Variable#1
        self.beta = nn.Parameter(torch.zeros(features))  #0
        self.eps = eps#这里self.eps是为了避免分母为0设置的

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)#均值,后面参数表示保持维度数量不变,因为对某一维度求均值后，维度数量会减1，TRUR表示保持维度不变,这样才能相减。
        std = x.std(-1, keepdim=True)#标准差
        print('x:',x)
        print('mean:',mean)
        print('std:',std)
        print('x-mean:',x-mean)

        result=self.gamma * (x - mean) / (std + self.eps) + self.beta
        print('result',result)
        return result
nomal=LayerNorm(2)#特征维度是2
x=np.random.randn(100,2)*100#批量是3
#x=np.asarray([[10,4],[100,90],[0,89],[67,45],[2,70],[-90,67],[43,-20]])
n=np.split(x,[1],axis=1)#先将数组按特征拆分，也就是按列拆分
pl.plot(np.hstack(n[0]),np.hstack(n[1]),'.')#然后再按行合并
pl.show()
# print(x)
# print(np.hstack(n[0]))


x=Variable(torch.Tensor(x))#批量是10
result=nomal(x)
result=result.data.numpy()*10000
result=np.split(result,[1],axis=1)
pl.plot(np.hstack(result[0]),np.hstack(result[0]),'.')
pl.show()
# print('result:',result)
