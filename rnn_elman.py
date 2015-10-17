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


if __name__ == '__main__':
    RNNSLU(100, 10, 1000, 50, 5)

    
    
