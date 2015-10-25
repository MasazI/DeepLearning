#coding: utf-8

from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

# データセット
datasets = {'imdb': (imdb.load_data(path='data/imdb/imdb.pkl'), imdb.prepare_data)}

# random number generators seeds for consistency
SEED = 123
np.random.seed(SEED)

# floatX型の配列の取得
def numpy_floatX(data):
    ''' 配列をtheanoのfloatX型へ変換する  '''
    return np.asarray(data, dtype=config.floatX)

# データセットの取得
def get_dataset(name):
    return datasets[name][0], datasets[name][1]

# モデルのzip
def zipp(params, tparams):
    ''' モデルをリロードするときは、GPU資源が必要  '''
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# モデルのunzip
def unzip(zipped):
    ''' モデルをpickleするときは、GPU資源が必要  '''
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# dropout
def dropout_layer(state_before, use_noise, trng):
    '''
        state_before: 前の状態 i.e. 前のレイヤー 
        use_noise: theanoの共有変数  switchのcondとして使う。0 or 1を指定し、1ならばift、0ならばiffを実行する
        trng: theano RandomSeedsオブジェクト
    '''
    # tensor.switchによるdropout実装
    # use_noiseが1のときは、前の状態にランダムなノイズを加える
    # use_noiseが0のときは、前の状態に単純に0.5を掛け、半分有効にする
    proj = T.switch(use_noise, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)
    return proj

# minibatch index
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    # arangeは0からnまでの配列をつくる
    idx_list = np.arange(n, dtype='int32')
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if(minibatch_start != n):
        # minibatchの開始indexがnでない場合、余り部分を最後に追加
        minibatches.append(idx_list[minibatch_start:])

    # zipはまとめてループできる関数であり、minibatchesの数でindexをつくり、minibatcheの配列をindexごとに収める
    return zip(range(len(minibatches)), minibatches)

# parameter名の作成
def _p(pp, name):
    return '%s_%s' % (pp, name)

# (lstm以外の)埋め込みや分類用の基本パラメータの初期化
def init_params(options):
    params = OrderedDict()
    
    # 埋め込み行列 (n_words:単語数)x(dim_proj:埋め込み空間の次元=hidden layerのユニット数)
    randn = np.random.rand(options['n_words'], options['dim_proj'])
    # 埋め込み行列 Wemb は、randn*0.01 を初期値に与える
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options, params, prefix=options['encoder'])

    # classifier
    # 分類行列 (dim_proj:埋め込み空間の次元=hidden layerのユニット数)x(ydim:分類クラス数)
    params['U'] = 0.01 * np.random.randn(options['dim_proj'], options['ydim']).astype(config.floatX)
    # バイアス(ydim:分類クラス数の長さのベクトル)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return params    

# パラメータのロード
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is no in the archive' % kk)
        params[kk] = pp[kk]
    return params

# tパラメータの初期化
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# 名前からレイヤーを取得
def get_layer(name):
    fns = layers[name]
    return fns


# テストコード
def test():
    ## test numpy floatX
    print theano.shared(numpy_floatX(0.)).eval()

    ## test get dataset
    get_dataset('imdb')

    ## test minibatch index
    get_minibatches_idx(10, 2, False) 

    ## test _p   
    print _p('lstm', 'W')

if __name__ == '__main__':
    test()
    
    


