#multiprocessing shared list queue
#解决共享buffer问题
import random
import time
import torch.multiprocessing as mp





def sample0(q,p):
    for i in  [1, 2, 3]:
        p.put(i)
        time.sleep(1)
        q.append(i)
    print('sample:',random.sample(list(q),2))
    print('end 0')


def sample1(q,p):
    for i in [4, 5, 6]:
        p.put(i)
        q.append(i)
    print('sample:',random.sample(list(q), 2))
    print('end 1')



from multiprocessing import Manager
manager=Manager()
# manager.Value('i',1)
# manager.Array('i',range(10))
# manager.dict()
# manager.lock()

q=manager.list([7,8,10])
p=mp.Queue()

t1=mp.Process(target=sample0,args=(q,p))
t2=mp.Process(target=sample1,args=(q,p))
t1.start()
t2.start()

t1.join()
t2.join()

# print(q[:])
# while True:
#     print(p.get())
print(type(q))
print(random.sample(list(q),4))