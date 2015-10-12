# coding: utf-8
import cPickle
import gzip
import os
import sys
import timeit

import numpy as np
import theano
import theano.tensor as T

# debug
import pylab

def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # 共有変数に格納して使うと、GPUのメモリ領域に保存される(その設定なら)なので高速
    def shared_dataset(data_xy, borrow=True):
        # data_xyをsplit
        data_x, data_y = data_xy

        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

        # ラベルはintなのでcast
        return shared_x, T.cast(shared_y, 'int32')

    ## 
    #train_set_x, train_set_y = train_set
    #pylab.figure()
    #for index in range(100):
    #    pylab.subplot(10, 10, index + 1)
    #    pylab.axis('off')
    #    pylab.imshow(train_set_x[index, :].reshape(28, 28), cmap=pylab.cm.gray, interpolation='nearest') 
    #pylab.show()
    #exit()

    # データを用途ごとに共有変数に格納
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # 行列にしてかえす
    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
 
    return rval

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        ## model
        ### wieght
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        ### bias
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        ### probabilities of y given x (input は x のミニバッチ)
        ### 出力はn_samples, n_outの行列
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        ### prediction 1dim
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        ### parameters
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        # 事例ごとの各クラスに分類される確率を表したのがself.py_y_given_x、[T.arange(y.shape[0), y]で事例の正解ラベルに分類される確率のみをスライスする
        # スライスされると、正解クラスに分類される確率のみを取得することになり、最大で1になる。符号を逆にすることで、最大1を最もlossの小さい状態としている。
        # オンラインで学習する場合は和で良いが、ミニバッチを利用する場合はmeanを使用する。
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])    

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='data/mnist.pkl.gz', batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size

    print '... building the model'

    # long scalar
    index = T.lscalar()

    # 教師画像のデータ
    x = T.matrix('x')

    # 教師ラベルのデータ
    y = T.ivector('y')

    ## まずはシンボルの定義
    # 分類器のシンボル
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

    # 損失のシンボル
    cost = classifier.negative_log_likelihood(y)

    # テスト用modelのfunction化
    print '...... test model'
    test_model = theano.function(
        inputs=[index], # 入力データ
        outputs=classifier.errors(y), # 損失関数
        givens={
            x:test_set_x[index*batch_size : (index + 1)*batch_size],
            y:test_set_y[index*batch_size : (index + 1)*batch_size]
        }
    )

    # バリデーション用modelのfunction化
    print '...... validation model'
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x:valid_set_x[index*batch_size : (index + 1)*batch_size],
            y:valid_set_y[index*batch_size : (index + 1)*batch_size]
        }
    )
    
    # 学習用の勾配計算のシンボル
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # 学習モデル更新計算のシンボル
    updates = [(classifier.W, classifier.W - learning_rate*g_W), (classifier.b, classifier.b - learning_rate*g_b)]

    # 学習用modelのfunction化
    print '...... train model'
    train_model = theano.function(
        inputs=[index],
        outputs=cost, #損失関数
        updates=updates, #モデル更新
        givens={
            x:train_set_x[index*batch_size : (index + 1)*batch_size],
            y:train_set_y[index*batch_size : (index + 1)*batch_size]
        }
    )

    ## 学習の開始
    print '... training the model'
    # 収束判定に利用するpatience
    patience = 5000
    patience_increase = 2
    improvement_threashold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()
    
    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # minibatchのindexを渡してfunctionを実行
            minibatch_avg_cost = train_model(minibatch_index)

            # 全体のiteration数 
            iter = (epoch - 1)*n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # validationのminiバッチごとのlossをもとめ、平均を計算
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f ' % (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
            
                if this_validation_loss < best_validation_loss:
                    # 今回のエラー率が最も良かったエラー率と近い場合、学習回数を増やす
                    if this_validation_loss < best_validation_loss * improvement_threashold:
                        patience = max(patience, iter*patience_increase)
                        print("*** iter %d / patience %d" % (iter, patience))

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print('epoch %i, minibatch %i/%i, test error %f ' % (epoch, minibatch_index+1, n_train_batches, test_score*100.))
                with open('model/best_model_lr_mnist.pkl', 'w') as f:
                    cPickle.dump(classifier, f)
            
            # patienceを超えればループは終了
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("optimizetion done. best valid score %f %%, test score %f %%" % (best_validation_loss * 100, test_score * 100.))
    print("epochs: %d, with %f epochs/sec" % (epoch, 1.0 * epoch / (end_time - start_time)))
   
def predict():
    classifier = cPickle.load(open('model/best_model_lr_mnist.pkl'))
             
    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)
    
    dataset='/Users/masai/source/theano/ml/data/mnist.pkl.gz' 
    datasets = load_data(dataset) 

    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    
    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set.")
    print predicted_values

    pylab.figure()
    for index in range(10):
        pylab.subplot(1, 10, index + 1) #行数、レス数、何番目のプロットか入力
        pylab.axis('off')
        pylab.title(predicted_values[index])
        pylab.imshow(test_set_x[index, :].reshape(28, 28), cmap=pylab.cm.gray, interpolation='nearest') 
    pylab.show()
    exit()


if __name__ == '__main__':
    sgd_optimization_mnist()
    predict()  
