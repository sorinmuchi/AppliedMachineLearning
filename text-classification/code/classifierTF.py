import numpy as np
import random
def split(X,Y,ratio=2/3): # Split into training and validation sets
    n=len(X)
    trainIndices=random.sample(range(0,n),int(n*ratio))
    testIndices=list(set(range(n))-set(trainIndices))
    # Split by train/test
    X_train=X[trainIndices,:]
    X_test=X[testIndices,:]
    Y_train=Y[trainIndices,:]
    Y_test=Y[testIndices,:]
    assert len(trainIndices)+len(testIndices)==n
    return [X_train,X_test,Y_train,Y_test]
def sync_shuffle(a, b): # syncronize shuffling of x and y
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p,:], b[p,:]
