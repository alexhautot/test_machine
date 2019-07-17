import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def forward_dense(weights, bias, input):
    #forward propergation of a dense layer, currently using sigmoid activation
    Z = np.dot(weights,input)+bias
    A = sigmoid(Z)
    return A

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
            print(i)
            prev_len = i
            count +=1
        else:
            parameters["b"+str(count)]=np.zeros([i])
            w = np.zeros([i,prev_len])
            parameters["w"+str(count)]=init_weights(w,epsilon)
            prev_len = i
            count+=1
    return parameters

def cost_CE(preds, y):
    #not presently working
    #-y*log p + (1 - y)log(1-p

    cost = -y*np.log(preds)#+(1-y)*np.log(1-preds)
    return cost

def to_one_hot(Y,num_char):
    #converts Y to an array of one hot vectors
    m = Y.shape[1]
    new_Y = np.zeros([m,num_char])
    for i in range (m):
        num = Y[0][i]
        new_Y [i][num]=1
    return new_Y