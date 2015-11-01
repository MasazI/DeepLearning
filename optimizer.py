# coding: utf-8
import theano
import theano.tensor as T


'''
optimization methods. 
sgd
rmprop
adadelta
'''

def sgd(lr, tparams, grads, x, mask, y, cost):
    '''
        sgd(stochastic gradient descent)
        lr:: learning rate
        tparams:: モデルパラメータ
        grads:: コストの勾配
        x:: モデルへの入力データ
        mask:: シーケンスmask
        y:: ターゲット
        cost:: cost関数
    '''
    # ミニバッチの勾配を含む共有変数
    gshared = [theano.shared(p.get_value()*0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    # 勾配と勾配の共有変数のシンボル
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    
    # ミニバッチの勾配を計算する関数、weightsの更新はしない
    # input に x, mask, y
    # outputs に cost
    # updatesに パラメータの勾配、勾配の共有変数のシンボル
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')

    # 勾配gと学習率lrによってpを更新するシンボル
    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]
    
    # 学習率によって重みを更新する関数
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    # 勾配計算の関数と重み更新の関数を返す
    return f_grad_shared, f_update
