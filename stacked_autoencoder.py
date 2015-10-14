# coding: utf-8
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from multilayer_perceptron import HiddenLayer
from autoencoder import dA
from logistic_regression import load_data, LogisticRegression

import os
import sys
import numpy as np
import timeit

from utils import tile_raster_images
import PIL.Image as Image

class SdA(object):
    """ stacked autoencoder """
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10, corruption_levels=[0.1, 0.1]):
        # 必要なレイヤー配列を定義
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        # 隠れ層の数
        self.n_layers = len(hidden_layers_sizes)
        
        # 隠れ層の数は1以上
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        # 画像データ
        self.x = T.matrix('x')
        
        # int型正解ラベルデータ
        self.y = T.ivector('y')

        for i in xrange(self.n_layers):
            if i == 0:
                # 最初の隠れ層の入力データの数は、入力層のユニット数
                input_size = n_ins
            else:
                # 2つ目以降の隠れ層の入力データの数は、ひとつ前の隠れ層のユニット数
                input_size = hidden_layers_sizes[i-1]
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            # 隠れ層
            sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i], activation=T.nnet.sigmoid)
           
            # 隠れ層のリストに追加
            self.sigmoid_layers.append(sigmoid_layer)

            # 隠れ層のWとb
            self.params.extend(sigmoid_layer.params)            

            # AutoEncoder
            dA_layer = dA(numpy_rng=numpy_rng, theano_rng=theano_rng, input=layer_input, n_visible=input_size, n_hidden=hidden_layers_sizes[i], W=sigmoid_layer.W, bhid=sigmoid_layer.b)
            
            # AutoEncoderのリストに追加
            self.dA_layers.append(dA_layer)

        # sigmlid_layresの最後のレイヤーを入力にする、hidden_layers_sizesの最後の層は入力のユニット数、出力ユニットの数はn_outs
        self.logLayer = LogisticRegression(input=self.sigmoid_layers[-1].output, n_in=hidden_layers_sizes[-1], n_out=n_outs)

        # LogisticRegression層のWとb
        self.params.extend(self.logLayer.params)

        # 正則化項は無しで良い
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # LogisticRegression層のエラーを使う
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        """ 各レイヤーのAutoEncoderによる学習 """
        # minibatchのindex
        index = T.lscalar('index')

        # ノイズ率
        corruption_level = T.scalar('corruption')
        # 学習率
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        # 事前学習のfunctionリスト
        pretrain_functions = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level, learning_rate)

            fn = theano.function(inputs=
                [
                    index, 
                    theano.Param(corruption_level, default=0.2), # Paramを使うとTensorの引数の名前で値を指定できる
                    theano.Param(learning_rate, default=0.1)
                ], 
                outputs=cost, 
                updates=updates, 
                givens={self.x: train_set_x[batch_begin: batch_end]}
            )
            # 事前学習の関数リストに各層のオートエンコーダのcost計算とパラメータupdatesのfunctionを追加
            pretrain_functions.append(fn)
        return pretrain_functions
    
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """ ネットワーク全体でfinetuning """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # まとめて計算するようのbatch
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

        index = T.lscalar('index')
        
        # logiticlayerの微分
        gparams = T.grad(self.finetune_cost, self.params)
        # ネットワークのパラメータ更新
        updates = [(param, param - gparam * learning_rate) for param, gparam in zip(self.params, gparams)]

        train_model = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates, givens={self.x: train_set_x[index*batch_size : (index+1)*batch_size], self.y: train_set_y[index*batch_size : (index+1)*batch_size]}, name='train')

        # minibatch index i testのエラースコアfunction
        test_score_i = theano.function(inputs=[index], outputs=self.errors, givens={self.x: test_set_x[index*batch_size : (index+1)*batch_size], self.y: test_set_y[index*batch_size : (index+1)*batch_size]}, name='test')

        # minibatch index i validateのエラースコアfunction
        valid_score_i = theano.function(inputs=[index], outputs=self.errors, givens={self.x: valid_set_x[index*batch_size : (index+1)*batch_size], self.y: valid_set_y[index*batch_size : (index+1)*batch_size]}, name='validate')

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

def optimize_stacked_autoencoder(n_ins=28*28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10, corruption_levels=[0.1, 0.2, 0.3], pretraining_epochs=30, pretrain_lr=0.001, training_epochs=1000, finetune_lr=0.1, dataset='data/mnist.pkl.gz', batch_size=1):
    """ 各事前学習のエポック数、事前学習の学習率、finetuneのエポック数、finetuneの学習率、学習データああセット、ミニバッチサイズ"""
    assert len(hidden_layers_sizes) == len(corruption_levels)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 教師バッチ数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    numpy_rng = np.random.RandomState(89677)
    
    print "building the model ..."

    sda = SdA(numpy_rng=numpy_rng, n_ins=n_ins, hidden_layers_sizes=hidden_layers_sizes, n_outs=n_outs)

    print "getting the pretraining functions ..."

    pretraining_functions = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

    print "pre-training the model ..."

    start_time = timeit.default_timer()

    # 層ごとにAutoEncode
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_functions[i](index=batch_index, corruption=corruption_levels[i], lr=pretrain_lr))
                print "Pre-training layer %i, epoch %d, batch %i/%i cost %f" % (i, epoch, batch_index, n_train_batches, np.mean(c))
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    print "The pretraining code for file %s ran for %.2fm" % (os.path.split(__file__)[1], training_time / 60.0)

    # AutoEncodeされたネットワークをfinetuningする関数を取得
    print "get the finetuning functions ..."
    train_model, validate_model, test_model = sda.build_finetune_functions(datasets=datasets, batch_size=batch_size, learning_rate=finetune_lr)

    print "fine-tuning the model ..."
    patience = 10 * n_train_batches
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    fp1 = open('log/SdA_validation_error.txt', 'w')
    fp2 = open('log/SdA_test_error.txt', 'w')

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

if __name__ == "__main__":
    optimize_stacked_autoencoder()
