import torch.nn as nn
import torch
from torch.autograd import Variable
embed=nn.Embedding(2,5)#nn.Embeding实际上就是根据词典和词向量维度产生一个向量矩阵，向量维度是词向量维度，向量个数是词典中词个数，这样每个词就对应了一个向量，这是没有经过调参的，随着整个网络的训练词向量也得到了训练
word_to_ix={'chen':[0],'hongming':[1],'ming':[2]}#词和词所对应的索引
chen_ix=word_to_ix['chen']
print(embed.weight)
chen_ix=Variable(torch.LongTensor(chen_ix))#一定转换为LongTensor
print(embed(chen_ix))




import torch.nn as NN
import torch
from torch.autograd import Variable
embeding=NN.Embedding(6,5)
word_to_index={'中':[0],'国':[1],'我':[2],'的':[3],'母':[4],'亲':[5]}
print(embeding.weight)
index=Variable(torch.LongTensor(word_to_index['中']))
print(embeding(index))
#同理句子向量
index=[word_to_index['中'],word_to_index['国'],word_to_index['我'],word_to_index['的'],word_to_index['母'],word_to_index['亲']]
index=Variable(torch.LongTensor(index))
print(embeding(index))
