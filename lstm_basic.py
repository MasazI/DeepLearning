# coding: utf-8
import numpy as np
import theano
import theano.tensor as T
from theano import config

import reber_grammer

import matplotlib.pyplot as plt

dtype = config.floatX

# logistic function for gate
sigma = lambda x: 1/(1 + T.exp(-x))

# activation function
act = T.tanh

# numpy random and random seed
SEED = 123
np.random.seed(SEED)

# sequences: x_t
# prior results: h_tm1, c_tm1
# non-sequences: W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y
def one_lstm_step(x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y):
    i_t = sigma(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
    f_t = sigma(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)

    c_t_prime = act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
    c_t = f_t * c_tm1 + i_t * c_t_prime

    o_t = sigma(theano.dot(x_t, W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, W_co) + b_o)

    h_t = o_t * act(c_t)

    y_t = sigma(theano.dot(h_t, W_hy) + b_y)
    return [h_t, c_t, y_t]


def ortho_weights(dim_x, dim_y):
    W = np.random.randn(dim_x, dim_y)
    u,s,v = np.linalg.svd(W)
    return W/s[0]

def lstm_train(n_in=7, n_hidden=10, n_i=10, n_c=10, n_o=10, n_f=10, n_y=7, nb_epochs=300, nb_train_examples=1000):
    '''
    # numbeer of input layer dim as embedded reber grammar (7bit vector)
        n_in = 7
    
    # number of hidden layer unit for gate
        n_hidden = 10
        n_i = 10
        n_c = 10
        n_o = 10
        n_f = 10
    
    # number of output layer dim (7bit vector)
        n_y = 7
    '''
    
    # 重みの初期化
    # 入力および出力ゲートは開くか閉じるを使う
    # 忘却ゲートは開いているべきある。（トレーニングのはじめから忘れないように）
    # biasの適当な初期化によって達成を試みている。
    W_xi = theano.shared(ortho_weights(n_in, n_i))
    W_hi = theano.shared(ortho_weights(n_hidden, n_i))
    W_ci = theano.shared(ortho_weights(n_c, n_i))
    b_i = theano.shared(np.cast[config.floatX](np.random.uniform(-0.5,0.5,size=n_i))) # 入力ゲートはランダムで良い
    
    W_xf = theano.shared(ortho_weights(n_in, n_f))
    W_hf = theano.shared(ortho_weights(n_hidden, n_f))
    W_cf = theano.shared(ortho_weights(n_c, n_f))
    b_f = theano.shared(np.cast[config.floatX](np.random.uniform(0,1,size=n_f))) # 忘却ゲートははじめ開いているべき(sigmoidを挟んだときに確実に0.5以上にする)
    
    W_xc = theano.shared(ortho_weights(n_in, n_c))
    W_hc = theano.shared(ortho_weights(n_hidden, n_c))
    b_c = theano.shared(np.zeros(n_c, dtype=config.floatX)) # メモリセルのバイアスは初期値0（閉じるでも開くでもない）
    
    W_xo = theano.shared(ortho_weights(n_in, n_o))
    W_ho = theano.shared(ortho_weights(n_hidden, n_o))
    W_co = theano.shared(ortho_weights(n_c, n_o))
    b_o = theano.shared(np.cast[config.floatX](np.random.uniform(-0.5,0.5,size=n_o))) # 出力ゲートはランダムで良い
    
    W_hy = theano.shared(ortho_weights(n_hidden, n_y))
    b_y = theano.shared(np.zeros(n_y, dtype=config.floatX))  # カテゴリー分類レイヤーのバイアスは初期値0
    
    c0 = theano.shared(np.zeros(n_c, dtype=config.floatX)) # メモリセルの初期入力
    h0 = T.tanh(c0) # 初期ct_prime
    
    params = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y, c0]
    
    # 初期時刻の入力ベクトルシンボル
    v = T.matrix(dtype=config.floatX)
    
    # ターゲット(教師データ)シンボル
    target = T.matrix(dtype=config.floatX)
    
    # recurrence
    [h_vals, _, y_vals], _ = theano.scan(fn=one_lstm_step, 
                                        #sequences = dict(input=v, taps=[0]), 
                                        sequences = v,
                                        outputs_info = [h0, c0, None],
                                        non_sequences = [W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, W_co, b_o, W_hy, b_y])
    
    # cost ここでは多クラス問題なのでクロスエントロピー
    cost = -T.mean(target * T.log(y_vals) + (1.-target)*T.log(1.-y_vals))
    
    # 学習率の共有変数
    lr = np.cast[config.floatX](.1)
    learning_rate = theano.shared(lr)
    
    # 各パラメータの勾配
    #gparams = T.grad(cost, params)
    
    gparams = []
    for param in params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # パラメータの更新 simple sgd
    updates = []
    for param, gparam in zip(params, gparams):
        updates.append((param, param - gparam * learning_rate))
    
    # 教師データの生成
    train_data = reber_grammer.get_n_embedded_examples(nb_train_examples)
    print 'train data length: ',len(train_data)    

    # lstm
    learn_rnn_fn = theano.function(inputs=[v, target], outputs=cost, updates=updates)
    train_errors = np.ndarray(nb_epochs)
    def train_rnn(train_data):
        for x in range(nb_epochs):
            error = 0.
            for j in range(len(train_data)):
                # train_dataからランダムに1つ事例を取得
                index = np.random.randint(0, len(train_data))
                # 入力ベクトルi, 教師ベクトルo
                i, o = train_data[index]
                #print 'train vector: ',i
                #print 'train target: ',o
                train_cost = learn_rnn_fn(i, o)
                error += train_cost
            # epochごとにerrorを出力
            print "epochs %i : %f"%(x, error)
            train_errors[x] = error

    train_rnn(train_data)

    plt.plot(np.arange(nb_epochs), train_errors, 'b-')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.ylim(0., 50)
    plt.show()
    print params

if __name__ == '__main__':
    # orthogonal weights test
    # print ortho_weights(5,5)

    # basic lstm train
    lstm_train() 
