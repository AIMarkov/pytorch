import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.functional as f
rnn=nn.RNN(input_size=20,hidden_size=50,num_layers=2)
x=Variable(torch.randn(100,32,20))
h0=Variable(torch.zeros(2,32,50))
output,h_n=rnn(x,h0)
print(output)