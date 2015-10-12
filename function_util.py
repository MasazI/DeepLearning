#coding: utf-8
import numpy as np
import theano
import theano.tensor as T

## Define Symbol
def sym_ReLU(data):
    return T.switch(data < 0, 0, data)

## Define function
def func_ReLU(data):
    x = T.fmatrix('x')
    s = T.switch(x < 0, 0, x)
    relu = theano.function(inputs=[x], outputs=s)
    return relu(data)

def func_sigmoid(data):
    ## define of x
    x = T.fmatrix('x')

    ## define of compute
    # s = T.nnet.sigmoid(x)でも同じ
    s = 1 / (1 + T.exp(-x))

    ## define of function 入力が行列でも要素ごとに計算される(for文などは使わずに、ベクトル化して一気に計算するのが良い)
    sigmoid = theano.function(inputs=[x], outputs=s)

    #return sigmoid([[0, 1], [-1, -2]])
    return sigmoid(data)

def func_softmax(data):
    ## define of x
    x = T.fmatrix('x')
    
    s = T.nnet.softmax(x)

    softmax = theano.function(inputs=[x], outputs=s)

    return softmax(data)

## Test
if __name__ == '__main__':
    #d = theano.shared(value=np.matrix([[0.1,0.2,0.3],[0.3,0.2,0.1],[0.1,0.2,0.3]]), name='d', borrow=True)
    d = [[-1.,2.,-3.],[4.,5.,6.],[7.,8.,9.]]
    rsigmoid = func_sigmoid(d)
    print rsigmoid

    ssoftmax = func_softmax(d)
    print ssoftmax

    srelu = func_ReLU(d)
    print srelu
