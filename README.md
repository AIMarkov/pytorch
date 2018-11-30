# pytorch
>>比较好的网站，多线程：https://ptorch.com

# CNN的原理图
## 卷积层
### 输入数据尺寸是W\*H\*D。D表示深度，也表示通道（channel）数，所以输出尺寸也可以是W\*H\*C。
### 四个超参数：滤波器数量K,滤波器尺寸F，滑动步长S，零填充数量P
### 由于参数共享，每个滤波器包含的权重数量F\*F\*D，卷积层共有F\*F\*D\*K和K个偏置量
### 关于卷积后的图片维度计算公式：(W-F+2P)/S+1，其中W表示输入数据的大小，F表示卷积层神经元的感受野磁村，S表示步长，P表示边界填充数量，比如输入是7\*7，filter是3\*3，step是1.pading数量是0，那么更据公式有(7-3+2\*0)/1+1=5,即输出是5\*5，若步长是2则输出是3\*3.
![](https://github.com/AIMarkov/pytorch/raw/master/image/CNN/2256672-19110dee0c54c0b2.gif)
![](https://github.com/AIMarkov/pytorch/raw/master/image/CNN/2256672-958f31b01695b085.gif)
## 池化层
### 输入数据尺寸为W\*H\*D
### 两个超参数池化层尺寸F和步长S
### 输出W1\*H1\*D1,其中W1=(W-F)/S+1,H1=(H-F)/S+1,D1=D

# LSTM的结构
![](https://github.com/AIMarkov/pytorch/raw/master/image/LSTM/2256672-7ea82e4f1ac6cd75.png)
![](https://github.com/AIMarkov/pytorch/raw/master/image/LSTM/2256672-b784d887bf693253.png)
