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

def forward_pass_test(parameters, X, arc):
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
    dz = A-Y.T
    dw = (1/m)*np.dot(dz,X.T)
    db = (1/m)*np.sum(dz)
    return dz, dw, db

def back_prop_tanh(dzh,A,w,Alower):
    #takes the activation,  returns the derivatives dw and db
    m=A.shape[0]
    dz = np.multiply(np.dot(w.T, dzh), 1 - np.power(A, 2))
    dw = (1 / m) * np.dot(dz, Alower.T)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
    return dz, dw, db

def test_backprop(cache, pred, parameters,arcitecture, X,y):
    #written to test a simple arcitecture, intended to extend beyondd that
    steps = len(arcitecture)
    grads={}
    dz, dw, db = back_prop_final(pred, y, cache[2])
    grads.update({"dz4": "dz", "dw4": dw, "db4": db})
    dz, dw, db = back_prop_tanh(dz, cache[2], parameters["w4"], cache[1])
    grads.update({"dz3": "dz", "dw3": dw, "db3": db})
    dz, dw, db = back_prop_tanh(dz, cache[1], parameters["w3"], cache[0])
    grads.update({"dz2": "dz", "dw2": dw, "db2": db})
    dz, dw, db = back_prop_tanh(dz, cache[0], parameters["w2"], X)
    grads.update({"dz1": "dz", "dw1": dw, "db1": db})
    return grads

def param_update(parameters, grads, learning_rate, arc):
    #updates the parameters and returns them
    steps = len(arc)
    updated_params={}
    for i in range(1,steps):
        w = parameters["w"+str(i)]
        b = parameters["b"+str(i)]
        w = w - learning_rate*grads["dw"+str(i)]
        b = b - learning_rate*grads["db"+str(i)]
        updated_params.update({"w"+str(i):w, "b"+str(i):b})
    return updated_params

def nn_model(X, y, arcitecture, num_iterations, learning_rate):
    parameters = build_weight(arcitecture, epsilon=0.1)
    last_cost = 1000000
    for i in range (num_iterations):
        pred, cache = forward_pass(parameters, X, arcitecture)
        pred = sigmoid(pred)
        cost = cost_squared_loss(pred,y)
        grads = test_backprop(cache, pred, parameters, arcitecture, X, y)
        parameters = param_update(parameters, grads, learning_rate, arcitecture)
        if cost > last_cost:
            learning_rate=learning_rate / 3
        last_cost = cost
        if i% 300 ==0:
            accuracy(pred, y)
            print (cost)
    return parameters

def nn_model_retrain(X, y, arcitecture, num_iterations, learning_rate, parameters):
    last_cost = 1000000
    for i in range (num_iterations):
        pred, cache = forward_pass(parameters, X, arcitecture)
        pred = sigmoid(pred)
        cost = cost_CE(pred,y)
        grads = test_backprop(cache, pred, parameters, arcitecture, X, y)
        parameters = param_update(parameters, grads, learning_rate, arcitecture)
        if cost > last_cost:
            learning_rate=learning_rate / 3
        last_cost = cost
        if i%1000 ==0:
            print("train")
            accuracy(pred, y)
            print (cost)
    return parameters

def accuracy(pred, y):

    test = pred >0.5
    test = test * 1
    correct = test == y.T
    accuracy = np.sum(correct)/y.shape[0]
    print("set accuracy ", accuracy)

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

def cost_squared_loss(pred, Y):
    m = Y.shape[0]
    cost = np.sum(np.square(pred-Y.T),-1)/m
    return cost
