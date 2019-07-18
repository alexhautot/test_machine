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

parameters = mltools.build_weight(arcitecture,epsilon)

m=18

x = np.random.randint(0,high=10,size=(arcitecture[0],m))
y = np.array([1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0])



pred, cache = mltools.forward_pass(parameters, x, arcitecture)
#cost = mltools.cost_CE(pred,y)

dz, dw, db = mltools.back_prop_final(pred, y, cache[1])
print(dz.shape)
print(dw.shape)
print(cache[1].shape)
print(parameters["w3"].shape)
dz, dw, db = mltools.back_prop_tanh(dz, cache[1],parameters["w3"],cache[0])
#back_prop_tanh and back_prop_final are functioning (not broken yet)

#working possibly as intended!

#testing forward_dense, X to be input vector
#b = np.random.randn(n_h)
#X = np.random.randn(n_f)

#a = mltools.forward_dense(w,b,X)
#print(a)


#testing cross entroy loss
#np.random.seed(seed=0)
#y = np.array([1,1,0,1])
#preds = np.random.randn(4)
#preds=mltools.sigmoid(preds)
#print(preds)
#cost = mltools.cost_CE(preds,y)
#lessons learnt: pass this through activation first, negative numbers will mess stuff up
#currently working (or at least hasn't broken yet)

#testing to_one_hot, working!
#np.random.seed(seed=0)
#Y = np.random.randint(0,high=9, size=(1,5))
#print(Y)
#Y = mltools.to_one_hot(Y,10)
#print(Y)

#loss = mltools.cost_CE(preds, y)
#print (loss)
