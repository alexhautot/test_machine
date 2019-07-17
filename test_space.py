import numpy as np

import mltools

epsilon = 0.000001

#testing out init_weights then forward_dense
#n_f = number of features, n_h = number of hidden units

n_f = 12
n_h = 8
w=np.ndarray([n_h,n_f])

w = mltools.init_weights(w,epsilon)

#testing build_weight
np.random.seed(seed=0)
arcitecture = [16,5,5,1]
parameters = mltools.build_weight(arcitecture)
print(parameters['b1'])
print(parameters['w1'])

#testing forward_dense, X to be input vector
#b = np.random.randn(n_h)
#X = np.random.randn(n_f)

#a = mltools.forward_dense(w,b,X)
#print(a)


#testing cross entroy loss
#y = np.array([1,1,0,1])
#preds = np.array([1,1,3,-1])
#testing to_one_hot, working!
#np.random.seed(seed=0)
#Y = np.random.randint(0,high=9, size=(1,5))
#print(Y)
#Y = mltools.to_one_hot(Y,10)
#print(Y)

#loss = mltools.cost_CE(preds, y)
#print (loss)
