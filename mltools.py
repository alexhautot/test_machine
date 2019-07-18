import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def forward_dense(weights, bias, input):
    #forward propergation of a dense layer, currently using sigmoid activation
    Z = np.dot(weights,input)+bias
    A = np.tanh(Z)
    return A

def forward_pass(parameters, X, arc):
    #takes the paramters and performs one forward pass, returns a, the prediction and a cache of activations
    steps = len(arc)-1
    a = X
    cache=[]
    for i in range(steps):
        w = parameters["w"+str(i+1)]
        b = parameters["b"+str(i+1)]
        a = forward_dense(w,b,a)
        cache.append(a)
    return a, cache

def back_prop_final(A,Y,X):
    #takes the final activation, ground truth and the previous activation, returns the derivatives dw and db
    m=Y.shape[0]
    dz = A-Y
    dw = (1/m)*np.dot(dz,X.T)
    db = (1/m)*np.sum(dz)
    return dz, dw, db

def back_prop_tanh(dzh,A,w,Alower):
    #takes the activation,  returns the derivatives dw and db
    m=A.shape[0]
    dz = np.multiply(np.dot(w.T, dzh), 1 - np.power(A, 2))
    dw = (1 / m) * np.dot(np.transpose(dz), Alower)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
    return dz, dw, db

def init_weights(w, epsilon):
    #randomly initilises weights of a layer to break symmetry
    dims = w.shape
    w=np.random.standard_normal(dims)*epsilon
    return w

def build_weight(arc, epsilon):
    #builds weights, using init_weights for w and np.zeros for b
    #returns parameters, a dictionary
    count=0
    parameters={}
    for i in arc:
        if count == 0:
            prev_len = i
            count +=1
        else:
            parameters["b"+str(count)]=np.zeros([i,1])
            w = np.zeros([i,prev_len])
            parameters["w"+str(count)]=init_weights(w,epsilon)
            prev_len = i
            count+=1
    return parameters

def cost_CE(preds, y):
    #takes a matrix of predictions and a ground truth matrix, outputs cost_CE
    #not presently working
    #-y*log p + (1 - y)log(1-p
    m = y.shape[0]
    cost = np.dot(np.log(preds),-y)+np.dot(np.log(1-preds),(y-1))
    cost = cost/m
    return cost

def to_one_hot(Y,num_char):
    #converts Y to an array of one hot vectors
    m = Y.shape[1]
    new_Y = np.zeros([m,num_char])
    for i in range (m):
        num = Y[0][i]
        new_Y [i][num]=1
    return new_Y
