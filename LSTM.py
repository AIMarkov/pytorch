import torch
import torch.nn as nn
from torch.autograd import Variable
class ex_actor_lstm(nn.Module):
    def __init__(self):
        super(ex_actor_lstm, self).__init__()
        self.scale=0.001
        self.batch=64
        self.lstm=nn.LSTM(input_size=20,hidden_size=200,num_layers=2,batch_first=True)
        self.linear=nn.Linear(200,8)
        # for weight in self.lstm.all_weights:
        #     nn.init.xavier_uniform(weight)
        nn.init.xavier_uniform(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform(self.lstm.weight_ih_l1)
        nn.init.xavier_uniform(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform(self.lstm.weight_hh_l1)
        nn.init.xavier_uniform(self.linear.weight.data)
    def forward(self, input):
        # h0=Variable(torch.zeros(2,self.batch,200))
        # c0=Variable(torch.zeros(2,self.batch,200))默认传入的隐藏状态ｈ０和Ｃ０都是０
        out,_=self.lstm(input)
        out=out[:,-1,:]#输入是序列，输出也是序列（每个时刻都有输出），这里取序列(序列长度为４)最后一个作为结果
        return self.linear(out)
lstm=ex_actor_lstm()
x=Variable(torch.zeros(64,4,20))
lstm(x)