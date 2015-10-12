# coding: utf-8
import os
import sys

import numpy as np
import theano
import theano.tensor as T
import timeit

import logistic_regression

class HiddenLayer(object):
    ## rmp: numpy.random.RandomState. a random number generator used to initilize weights
    ## input: theano.tensor.dmatrix. a symbolic tenso of shape(n_examples, n_in)
    ## n_in: int dim of input
    ## n_out: int num of hidden units
    ## activation: theano.Op or function activation.
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        if W is None:
            #Wの初期化
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6./(n_in + n_out)),
                    high=np.sqrt(6./(n_in + n_out)),
                    size=(n_in, n_out)),
                    dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                # sigmoidの場合は4倍する
                W_values *= 4
            # 学習する変数を共有型にする
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # bの初期化(n_out行)
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            # 学習する変数を共有型にする
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.input = input

        #隠れ層からの出力を計算
        #入力と重みの内積とバイアス
        lin_output = T.dot(input, self.W) + self.b

        #出力を計算
        #activationがNoneの場合(activationを挟まない線形変換の場合)
        if activation is None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)

        # parameters        
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # 隠れ層
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        # 出力層
        self.logRegressionLayer = logistic_regression.LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

        # L1正則化項
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        # L2正則化
        self.L2_sqr = ((self.hiddenLayer.W) ** 2).sum() + ((self.logRegressionLayer.W) ** 2).sum()

        # loss（出力層にのみ依存するのでロジスティック回帰と同じで良い）
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # 誤差計算シンボル
        self.errors = self.logRegressionLayer.errors

        # パラメータ
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # self tracking input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='data/mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = logistic_regression.load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # ミニバッチのindex
    index = T.lscalar()
    # 事例ベクトルx
    x = T.matrix('x')
    # int型の1次元ベクトル
    y = T.ivector('y')

    # ランダム変数
    rng = np.random.RandomState(1234)

    # MLPの構築
    classifier = MLP(rng=rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10)    

    # cost関数のシンボル 対数尤度と正則化項
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # ミニバッチごとのエラー率を計算するシンボル(test)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index+1)*batch_size],
            y: test_set_y[index * batch_size: (index+1)*batch_size]
        })

    # ミニバッチごとのエラー率を計算するシンボル(validation)
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index+1)*batch_size],
            y: valid_set_y[index * batch_size: (index+1)*batch_size]
        })
    
    # 勾配の計算 back propagation
    # gparamsに格納した変数でコストを偏微分する
    gparams = [T.grad(cost, param) for param in classifier.params]

    # パラメータの更新式のシンボル(複数の更新式を定義するときは配列にする)
    # classifierのparamとgparamsを同時にループ、paramsとその微分gparamsを使ったパラメータの更新式
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # 学習モデルでは、updatesに更新シンボルを入れてやれば良い
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index+1)*batch_size],
            y: train_set_y[index * batch_size: (index+1)*batch_size]
        })

    print '... training'
    patience = 10000
    patience_increase = 2
    improvement_threashold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                ## validationのindexをvalidationのエラー率を計算するfunctionに渡し、配列としてかえす
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                # 平均してscoreにする
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f ' % (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
            
                if this_validation_loss < best_validation_loss:
                    if(this_validation_loss < best_validation_loss * improvement_threashold):
                        patience = max(patience, iter*patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    ## testのindex をtestのエラー率を計算するfunctionに渡し、配列として渡す
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    
                    ## 平均してscoreにする
                    test_score = np.mean(test_losses)
                    ## 
                    print('epoch %i, minibatch %i/%i, test error %f ' % (epoch, minibatch_index+1, n_train_batches, test_score*100.))

            if patience <= iter:
                done_looping = True
                break
            
    end_time = timeit.default_timer()
    print(('optimization complete. Best validation score of %f obtained at iteration %i, with test performance %f') % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr,('This code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time)/60.))

if __name__ == '__main__':
    test_mlp()
