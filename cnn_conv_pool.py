#coding: utf-8
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from logistic_regression import load_data, LogisticRegression
from multilayer_perceptron import HiddenLayer
import function_util as func

import numpy as np
import timeit

import os
import sys

# 畳込み層
class ConvLayer(object):
    def __init__(self, rng, input, image_shape, filter_shape):
        # 入力のチャンネル数とフィルタを定義するときに指定する入力のチャンネル数の一致を確認
        assert image_shape[1] == filter_shape[1]
        
        # 入力の保存
        self.input = input

        # 黒魔術的なfilterの初期化
        # フィルターマップ数 * フィルターのheight * フィルターのwidth (prodはnpの配列要素全部の掛け算)
        fan_in = np.prod(filter_shape[1:])
        # 出力特徴Map数 * フィルターのheight * フィルターのwidth
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        # filterの定義
        W_bound = np.sqrt(6./ (fan_in + fan_out))
        # ランダムな値を割り振る
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

        # biasの定義
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # conv
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)

        # biasとactivate function
        # poolingの結果にbias項として加える(全項に追加)
        # biasは1dimのvectorなので(1, n_filters, 1, 1)にreshapeする
        # biasを加えたら、activate function(ここではtanh)を適用する
        self.output = func.sym_ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # パラメータの保存
        self.params = [self.W, self.b]

        # 入力の保存
        self.input = input

# プーリング層
class PoolLayer(object):
    def __init__(self, input, poolsize=(2,2)):
        # pooling
        self.input = input
        pooled_out = downsample.max_pool_2d(input=input, ds=poolsize, ignore_border=True)
        self.output = pooled_out

def optimize_cnn_lenet(learning_rate=0.01, n_epochs=200, dataset='data/mnist.pkl.gz', batch_size=500, n_hidden=500, nkerns=[20, 50], rng=np.random.RandomState(23455)):
    print '... load training set'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # ミニバッチのindex
    index = T.lscalar()

    # dataシンボル
    x = T.matrix('x')
    # labelシンボル
    y = T.ivector('y')

    print '... building the model'
    # LeNetConvPoolLayerと矛盾が起きないように、(batch_size, 28*28)にラスタ化された行列を4DTensorにリシェイプする
    # 追加した1はチャンネル数
    # ここではグレイスケール画像なのでチャンネル数は1
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # filterのnkerns[0]は20
    layer0 = ConvLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28), filter_shape=(nkerns[0], 1, 5, 5))
    
    layer1 = PoolLayer(layer0.output, poolsize=(2, 2))

    # filterのnkerns[1]は50
    layer2 = ConvLayer(rng, input=layer1.output, image_shape=(batch_size, nkerns[0], 12, 12), filter_shape=(nkerns[1], nkerns[0], 5, 5))

    layer3 = PoolLayer(layer2.output, poolsize=(2, 2))

    # layer2_input
    # layer1の出力は4x4ピクセルの画像が50チャンネル分4次元Tensorで出力されるが、多層パーセプトロンの入力にそのまま使えない
    # 4x4x50=800次元のベクトルに変換する(batch_size, 50, 4, 4)から(batch_size, 800)にする
    layer4_input = layer3.output.flatten(2)

    # 500ユニットの隠れレイヤー
    # layer2_inputで作成した入力ベクトルのサイズ=n_in
    layer4 = HiddenLayer(rng, input=layer4_input, n_in=nkerns[1]*4*4, n_out=n_hidden, activation=T.tanh)

    # 出力は500ユニット
    layer5 = LogisticRegression(input=layer4.output, n_in=n_hidden, n_out=10)
    
    # cost(普通の多層パーセプトロンは正則化項が必要だが、CNNは構造自体で正則化の効果を含んでいる)
    cost = layer5.negative_log_likelihood(y)

    # testモデル
    # 入力indexからgivensによって計算した値を使ってlayer3.errorsを計算する
    test_model = theano.function([index], layer5.errors(y), givens={x:test_set_x[index*batch_size : (index + 1)*batch_size], y: test_set_y[index*batch_size : (index + 1)*batch_size]})
    
    # validationモデル
    validate_model = theano.function([index], layer5.errors(y), givens={x:valid_set_x[index*batch_size : (index + 1)*batch_size], y: valid_set_y[index*batch_size : (index + 1)*batch_size]})

    # 微分用のパラメータ(pooling層にはパラメータがない)
    params = layer5.params + layer4.params + layer2.params + layer0.params

    # コスト関数パラメータについてのの微分
    grads = T.grad(cost, params)

    # パラメータの更新
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

    # trainモデル
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x[index*batch_size : (index + 1)*batch_size], y:train_set_y[index*batch_size : (index+1)*batch_size]})

    # optimize
    print "train model ..."
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    fp1 = open('log/lenet_validation_error.txt', 'w')
    fp2 = open('log/lenet_test_error.txt', 'w')

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
                fp1.write("%d\t%f\n" % (epoch, this_validation_loss*100))         

                if this_validation_loss < best_validation_loss:
                    if(this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter*patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    ## testのindex をtestのエラー率を計算するfunctionに渡し、配列として渡す
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    
                    ## 平均してscoreにする
                    test_score = np.mean(test_losses)
                    ## 
                    print('epoch %i, minibatch %i/%i, test error %f ' % (epoch, minibatch_index+1, n_train_batches, test_score*100.))
                    fp2.write("%d\t%f\n" % (epoch, test_score*100))
            if patience <= iter:
                done_looping = True
                break
    fp1.close()
    fp2.close()        
    end_time = timeit.default_timer()
    print(('optimization complete. Best validation score of %f obtained at iteration %i, with test performance %f') % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr,('This code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time)/60.))

    import cPickle
    cPickle.dump(layer0, open("model/cnn_layer0.pkl", "wb"))
    cPickle.dump(layer2, open("model/cnn_layer2.pkl", "wb"))
    cPickle.dump(layer4, open("model/cnn_layer4.pkl", "wb"))
    cPickle.dump(layer5, open("model/cnn_layer5.pkl", "wb"))

if __name__ == '__main__':
    optimize_cnn_lenet()
    
