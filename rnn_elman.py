#coding: utf-8
from collections import OrderedDict
import copy
import cPickle
import os
import urllib
import random
import stat
import subprocess
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

import cPickle
import pdb

import rnn
import rnn_tools

# python の最大再帰数の設定（Cスタックがオーバーフローしてクラッシュすることを防ぐ）
sys.setrecursionlimit(1500)

class RNNSLU(object):
    """ elman rnn """
    def __init__(self, nh, nc, ne, de, cs):
        """ nh: 隠れ層の次元
            nc: 分類クラス数
            ne: ワード埋め込み空間のボキャブラリー数
            de: ワード埋め込み空間の次元数
            cs: コンテキストウィンドウのサイズ
        """

        ## 最適化するモデルのパラメータを定義する
        
        # ワード埋め込み空間
        # ボキャブラリーの個数*埋め込み空間の次元数 ワード埋め込み空間(+1はt+1に相当する行をパディングで追加)
        self.emb = theano.shared(name='embeddings', value=0.2*np.random.uniform(-1.0,1.0,(ne+1, de)).astype(theano.config.floatX))
        
        # 入力ベクトルから隠れ層の重み
        # (ワード埋め込み次元数*コンテキストウィンドウサイズ) * 隠れ層の次元 の行列
        self.wx = theano.shared(name='wx', value=0.2*np.random.uniform(-1.0,1.0,(de*cs, nh)).astype(theano.config.floatX))

        # 隠れ層から隠れ層への重み
        self.wh = theano.shared(name='wx', value=0.2*np.random.uniform(-1.0,1.0,(nh, nh)).astype(theano.config.floatX))

        # 隠れ層から出力層への重み
        self.w = theano.shared(name='wx', value=0.2*np.random.uniform(-1.0,1.0,(nh, nc)).astype(theano.config.floatX))

        # 隠れ層のbias
        self.bh = theano.shared(name='bh', value=np.zeros(nh, dtype=theano.config.floatX))

        # 出力層のbias
        self.b = theano.shared(name='b', value=np.zeros(nc, dtype=theano.config.floatX))

        # はじめの隠れ層へのbias
        self.h0 = theano.shared(name='h0', value=np.zeros(nh, dtype=theano.config.floatX))

        # 微分用にパラメータを配列に入れておく
        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        # 入力の行列シンボル コンテキストウィンドウを行にもつセンテンスの行列
        idxs = T.imatrix()

        # 入力データを入力データを埋め込み空間に写像した行列
        # 行: idxs.shape[0]は入力データの行数、列：(埋め込み空間の次元数*コンテキストウィンドウのサイズ　)
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))

        # ラベルデータ
        y = T.iscalar('y')
        y_sentence = T.ivector('y_sentence')

        # 伝播処理
        def recurrence(x_t, h_tm1):
            """
                x_t:: x of t
                h_tm1:: h of t-1
            """
            # 入力層と前の隠れ層からのデータ
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            # 隠れ層から出力するデータ
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t] # h_t(h of t), s_t(s of t)

        # scan.Iterator(fnを入力の各要素に対して適用する) outputs_infoはloop処理の初期値であり最初のrecurrenceへの引数。
        # x_t(x of t)に[self.h0, None]の配列が渡る
        # n_stepsは繰り返しの回数であり、xのshape[0]は入力データのコンテキストウィンドウの数と一致する
        # i.e. 入力データのコンテキストウィンドウの数だけ繰り返し処理する。各行には、コンテキストウィンドウのデータを埋め込み空間に写像したベクトルが入っているため。
        # ここではコンテキストウィンドウごとに、埋め込み空間に写像したベクトルに対して関数fnを適用するということになる
        # recurrenceには2つの引数がある。scan.Iterationの場合、第２引数は前回の結果をそのまま引き継ぐので、h_tm1にはh_tが入る。s_tは出力されるデータなので再利用しない。
        # _ にはscan.OrderedUpdatesが得られる。
        # _ は受け取らないとエラーになる。error: Outputs must be theano Variable or Out instances. Received OrderedUpdates()
        # _ は、繰り返し処理の順序有りアップデート処理をもっており、途中でストップする際などに用いることができる。_に明示的に条件を指定シない場合はn_stepsで止まるまで処理を行う
        [h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.h0, None], n_steps=x.shape[0])

        print _ 

if __name__ == '__main__':
    RNNSLU(100, 10, 1000, 50, 5)

    
    
