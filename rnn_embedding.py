# coding: utf-8

import theano
import theano.tensor as T
import numpy as np

from rnn_tools import contextwin

def embedding_test():
    # nv :: ボキャブラリーの個数
    # de :: 埋め込み空間の次元
    # cs :: コンテキストウィンドウサイズ
    nv, de, cs = 1000, 50, 5

    # 行列 (ボキャブラリーの個数+1)x(埋め込み空間の次元) - この単語の埋め込み空間も学習パラメータの1つ
    embeddings = theano.shared(0.2*np.random.uniform(-1.0, 1.0, (nv+1, de)).astype(theano.config.floatX))

    print embeddings.eval().shape
    print embeddings.eval()

    # 入力の行列シンボル=コンテキストウィンドウを行にもつセンテンスの行列
    idxs = T.imatrix()
    
    # 行：idxs.shape[0]は入力データの行数、列: (埋め込み空間の次元)*(コンテキストウィンドウサイズ)
    x = embeddings[idxs].reshape((idxs.shape[0], de*cs))

    sample = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    cwins = contextwin(sample, cs)
    print cwins

    # 関数化
    f = theano.function(inputs=[idxs], outputs=x)
    
    # サンプルのセンテンスをコンテンツウィンドウ処理した行列を、埋め込む空間に写像する
    a_embedded = f(cwins)

    print a_embedded.shape
    print a_embedded

if __name__ == '__main__':
    embedding_test()
