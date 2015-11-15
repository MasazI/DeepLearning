#encoding: utf-8

from __future__ import unicode_literals
from collections import defaultdict

import numpy as np
np.random.seed(1)

import networkx as nx
import matplotlib.pyplot as plt

# Markov Random Filed
class MRF(object):
    def __init__(self, input, theta=0.3, threshold=0.1):
        self.input = input
        # 入力画像のサイズ
        self.shape = self.input.shape
    
        # ノイズを含む観測値(visible) 可視変数
        self.visible = nx.grid_2d_graph(self.shape[0], self.shape[1])

        # ノイズを除去した推測値(hidden) 隠れ(潜在)変数    
        self.hidden = nx.grid_2d_graph(self.shape[0], self.shape[1])

        for n in self.nodes():
            # 入力画像の値を観測値ノードのvalueにセットする
            # inputは画像の2次元表現を2次元配列で保持
            self.visible[n]['value'] = self.input[n[0], n[1]]

            # 潜在変数が隣接ノードから受け取るメッセージを辞書として保存
            # {ノード座標: 各値{0, 1 をとる確率}
            # f()でarray[1.0, 1.0]を返す
            f = lambda: np.array([1.0, 1.0])
            # dictにlambdaを入れておいて、各要素でarrayを生成
            self.hidden[n]['messages'] = defaultdict(f)

        self.theta = theta
        self.threshold = threshold

    def nodes(self):
        # nodeの座標(行番号、列番号)のtupleを生成
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield(r, c)

def test_networkx():
    g = nx.Graph()
    g.add_node((0,0))
    g.add_node((0,1))
    g.add_edge((0,0), (0,1))

    nx.draw_networkx(g, node_color='w')
    #x軸の長さは不要
    plt.gca().xaxis.set_visible(False)
    #y軸の長さは不要
    plt.gca().yaxis.set_visible(False)
    plt.show()

def test_2d_networkx():
    nw = nx.grid_2d_graph(3, 3)
    # 辞書は{ノードのキー: (x座標、y座標)} 座標は左上が 0, 0 とした第4象限
    pos = {n: (n[0], -n[1]) for n in nw.nodes()}

    print pos

    nx.draw_networkx(nw, pos=pos, node_color='w')
     #x軸の長さは不要
    plt.gca().xaxis.set_visible(False)
    #y軸の長さは不要
    plt.gca().yaxis.set_visible(False)
    plt.show()

def test():
    test_networkx()
    test_2d_networkx()


if __name__ == '__main__':
    test()
