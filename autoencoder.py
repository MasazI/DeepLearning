# coding: utf-8
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import os
import sys
import numpy as np
import time

from utils import tile_raster_images
import PIL.Image as Image

from logistic_regression import load_data

class dA(object):
    ## theano_rngはTheanoの乱数発生器、n_visibleは入力データの次元、n_hiddenが隠れ層の次元
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # Wの共有変数
        if not W:
            # Wの行列はn_visible * n_hiddenだが、minibatch単位の計算を行うため、input * W になるので n_visible * n_hidden でシェイプを定義
            initial_W = np.asarray(numpy_rng.uniform(low=-4*np.sqrt(6./(n_hidden+n_visible)), high=4*np.sqrt(6./(n_hidden+n_visible)), size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)
        
        # 0で入力層のbiasを初期化した共有変数
        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), borrow=True)    
        # 0で隠れ層のbiasを初期化した共有変数
        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), borrow=True)

        # 順方向の重みW
        self.W = W

        # bは隠れ層のbias
        self.b = bhid

        # b_prime(bのチルダ)は入力層のbiasと同じ
        self.b_prime = bvis

        # 逆方向の重み
        # tied weights, weght tying
        self.W_prime = self.W.T

        # rng
        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        # 最適化パラメータ
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """ ノイズ画像の生成(binomial) """
        return self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)*input

    def get_hidden_values(self, input):
        """ 入力層の値を隠れ層の値に変換  """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """ 隠れ層の値を入力層の値に変換  """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ コストを計算してパラメータをアップデート """
        # xにnoiseを付加する(denoise)
        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        # 入力を変換
        # denoise
        y = self.get_hidden_values(tilde_x)
        # normal
        # y = self.get_hidden_values(self.x)

        # 変換した値を逆変換で元に戻して再構成
        z = self.get_reconstructed_input(y)

        # 元の入力と再構成した入力との交差エントロピー
        # xが0か1の2値を取る場合のエントロピーのはずだが、画像x
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # 誤差関数の微分
        gparams = T.grad(cost, self.params)

        # 更新式のシンボル
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
    
        return cost, updates

def test_autoencoder():
    learning_rate = 0.1
    training_epochs = 30
    batch_size = 20

    datasets = load_data('data/mnist.pkl.gz')

    train_set_x = datasets[0][0]

    # ミニバッチの数(教師データをbatch数で割るだけ)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # ミニバッチのindexシンボル
    index = T.lscalar()

    # ミニバッチの学習データシンボル
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # autoencoder モデル
    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28*28, n_hidden=500)

    # コスト関数と更新式のシンボル
    cost, updates = da.get_cost_updates(corruption_level=0.0, learning_rate=learning_rate)

    # trainingの関数
    train_da = theano.function([index], cost, updates=updates, givens={
            x : train_set_x[index*batch_size : (index+1)*batch_size]
        })

    fp = open("log/ae_cost.txt", "w")

    # training
    start_time = time.clock()
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d, cost ' % epoch, np.mean(c)
        fp.write('%d\t%f\n' % (epoch, np.mean(c)))

    end_time = time.clock()

    training_time = (end_time - start_time)

    fp.close()

    print "The no corruption code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((training_time / 60.0))
    
    image = Image.fromarray(tile_raster_images(
    X=da.W.get_value(borrow=True).T,
    img_shape=(28, 28), tile_shape=(10, 10),
    tile_spacing=(1, 1)))
    image.save('log/dae_filters_corruption_00.png')

if __name__ == "__main__":
    test_autoencoder()
