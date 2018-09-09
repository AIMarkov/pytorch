import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F

#官方文档,单层网络
rnn = nn.LSTMCell(10, 20)#输入是10,隐层是２０
input = Variable(torch.randn(6, 3, 10))#序列长度６，批量３，特征１０
print(input[0])
hx = Variable(torch.randn(3, 20))#批量３，隐层２０
cx = Variable(torch.randn(3, 20))
output = []
for i in range(6):#由于序列长度是６，所以循环６次
   hx, cx = rnn(input[i], (hx, cx))#每次输出有两个隐层状态，ｈｘ作为每个时间步结果，最后一个时间步的ｈｘ作为最终结果
   output.append(hx)


#多层网络
class LSTM(nn.Module):#
    def __init__(self):
        super(LSTM,self).__init__()
        self.ih0=nn.Linear(20,50)#第一输入层
        self.lstm0=nn.LSTMCell(50,50)#第一层ＬＳＴＭ
        self.ih1=nn.Linear(50,50)#第二输入层
        self.lstm1=nn.LSTMCell(50,50)#第二层ＬＳＴＭ

    def forward(self, input,h0,c0,h1,c1):
        x=self.ih0(input)
        h0,c0=self.lstm0(x,(h0,c0))
        x=self.ih1(h0)
        h1,c1=self.lstm1(x,(h1,c1))
        return  h0,c0,h1,c1
h0=Variable(torch.zeros(3,50))
c0=Variable(torch.zeros(3,50))
h1=Variable(torch.zeros(3,50))
c1=Variable(torch.zeros(3,50))
x=Variable(torch.randn(6,3,20))
lstm=LSTM()
for i in range(6):
    h0, c0, h1, c1=lstm(x[i],h0,c0,h1,c1)
    print(h1)#最后一层输出才是最终输出




