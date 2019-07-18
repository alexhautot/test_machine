import numpy as np

import h5py

import mltools

import mldata

X, Y, X_test, Y_test =  mldata.get_training_testing_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

def train_accuracy(parameters, X_test, Y_test, arcitecture):
    pred, cache = mltools.forward_pass(parameters, X_test, arcitecture)
    pred = mltools.sigmoid(pred)
    mltools.accuracy(pred, Y_test)


X=X.T
X_test=X_test.T


epsilon = 0.000001


np.random.seed(seed=10)
arcitecture = [12288,10,10,10,1]


#dz, dw, db = mltools.back_prop_final(pred, y, cache[1])
#print(dz.shape)
#print(dw.shape)
#print(cache[1].shape)
#print(parameters["w3"].shape)
#dz, dw, db = mltools.back_prop_tanh(dz, cache[1],parameters["w3"],cache[0])
#back_prop_tanh and back_prop_final are functioning (not broken yet)

#working possibly as intended!
#grads = test_backprop(cache, pred, parameters, arcitecture, x, y)

parameters = mltools.nn_model(X,Y, arcitecture, 15000, 0.01)
print("test")
train_accuracy(parameters, X_test, Y_test, arcitecture)

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
