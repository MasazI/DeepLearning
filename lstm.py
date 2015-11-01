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

# orthogonal initialization
# Wの特異値分解
def ortho_weight(ndim):
    # Wはランダムに取得
    W = np.random.randn(ndim, ndim)

    # ランダムに作成したWを特異値分解
    # np.linalgはlinear algebra
    # svdはsinglar value decomposition i.e. 特異値分解を行う関数
    # the matrix a as u * np.diag(s) * v
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

# lstm パラメータの初期化
def param_init_lstm(options, params, prefix='lstm'):
    '''
        options:: options[dim_proj]を利用する。dim_projはワード埋め込み空間の次元
        params:: paramsは重みのdict
        prefix:: 重みをparamsに格納する際に用いるprefix
    
        concatenateは行列の結合
    '''
    W = np.concatenate([ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W

    U = np.concatenate([ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj']), ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U

    b = np.zeros((4 * options['dim_proj'],))
    
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # state of below(prior) 以前の状態を取得
    nsteps = state_below.shape[0]

    # 以前の状態の次元が3の場合
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # maskはNoneではだめ
    assert mask is not None
    
    def _slice(_x, n, dim):
        '''
            _x:: 時刻tにおける前回の出力ベクトルh(t-1) と 行列U の積
            n:: input, forget, output, c を表すindex
            dim:: 各行列の次元
        '''
        # xが3次元の場合
        if _x.ndim == 3:
            # 複数サンプルを同時に計算している場合
            # 全サンプル分を切り出す
            return _x[:, :, n*dim : (n+1)*dim]
        # xが2次元の場合
        # 1サンプルごとに計算している場合
        return _x[:, n*dim : (n+1)*dim]

    def _step(m_, x_, h_, c_):
        '''
            m_:: mask
            x_:: state_below
            h_:: 前回の計算結果h
            c_:: 前回の計算結果c

            tにおけるm_とx_を利用してうまく計算して、t+1で利用するhとcを生成
        '''

        # 前回の出力ベクトルh_ = h(t-1) と行列Uの積
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        # x_を足す
        preact += x_

        # weights of input gate
        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))

        # weights of forget gate
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))

        # weights of output gate
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))

        # weights of c
        c = tensor.tanh(_slice(preact, 3, options[('dim_proj')]))

        # 入力cに入力ゲートをかけて、前回のメモリセルデータと忘却ゲートの積を足して、現時刻のメモリセルデータを生成
        c = f * c_ + i * c
        # m_[:, None]はm_に各行の間に1行ずつ空の行を追加する
        # 1. - m_ はmの各要素eを1 - eに置き換える
        # これはベクトルの演算と同じ意味あい
        # cベクトルとc_ベクトルをm_対1-m_の非で掛けあわせる
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        # 出力データはメモリセルのtanhに出力ゲートを掛けあわせる
        h = o * tanh(c)
        # hベクトルとh_ベクトルをm_対1_m_の非で掛けあわせる
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # 前回の状態
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])

    # ワード埋め込み空間の次元数
    dim_proj = options['dim_proj']

    # scanでrecurrent
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below], # mask = m_, state_below = x_
                                outputs_info=[T.alloc(numpy_floatX(0.), n_samples, dim_proj), T.alloc(numpy_floatX(0.), n_samples, dim_proj)], # (事例数*ワード埋め込み空間)次元の0で初期化された配列(h_とc_の初期値)
                                name=_p(prefix, '_layers'), # 名前はprefix_layers でprefixはlstmなので、lstm_layers
                                n_steps=nsteps)

    # rvalの最初の1つを返す
    return rval[0]

# lstmのパラメータとそのレイヤー
# 分類する場合は、noraml neural networkをlstmの後ろにおく
layers = {'lstm': (param_init_lstm, lstm_layer)}

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

    ## test init lstm parameters
    options = {'dim_proj':10}
    params = {}
    param_init_lstm(options, params)


if __name__ == '__main__':
    test()
    
    


