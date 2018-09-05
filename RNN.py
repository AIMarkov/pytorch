import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.functional as f
class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()
        self.num_layers=2
        self.hidden_size=10
        # self.model=nn.Sequential(
        #     nn.RNN(input_size=20,hidden_size=50,num_layers=2,batch_first=True)
        # )
        # self.model = nn.Sequential(
        #     nn.RNN(input_size=20, hidden_size=50, num_layers=2)
        # )
        self.rnn=nn.RNN(input_size=20, hidden_size=50, num_layers=2)
        #不能放在盒子nn.Sequential里，只能单独定义
    def forward(self, input):
        # h0 = Variable(torch.zeros(self.num_layers, input.size(1), self.hidden_size))
        h0=Variable(torch.zeros(2,32,50))#hidden state (hidden_layers,bach_size,hidden_size)
        # output,h_n=self.model(input,h0)
        output, h_n = self.rnn(input, h0)
        return output,h_n
Rnn1=Rnn()
# x=np.array([[[2,1],[3,4],[5,6]],[[2,2],[6,4],[2,6]]])
# x=x.reshape([2,3,2])
# x=torch.FloatTensor(x)
# x=Variable(x)
x=Variable(torch.randn(100,32,20))
print(x)
output,h_n=Rnn1(x)
print('output:',output)
print('h_n:',h_n)
