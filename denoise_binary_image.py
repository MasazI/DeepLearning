#encoding: utf-8

from __future__ import unicode_literals
from collections import defaultdict

import numpy as np
np.random.seed(1)

import networkx as nx
import matplotlib.pyplot as plt

import cPickle
import gzip
import pylab

# Markov Random Filed
class MRF(object):
    def __init__(self, input, theta=0.3, threshold=0.1, debug=False):
        self.input = input
        # 入力画像のサイズ
        self.shape = self.input.shape
    
        # ノイズを含む観測値(visible) 可視変数
        self.visible = nx.grid_2d_graph(self.shape[0], self.shape[1])

        # ノイズを除去した推測値(hidden) 隠れ(潜在)変数    
        self.hidden = nx.grid_2d_graph(self.shape[0], self.shape[1])

        # nはnodeの座標(行、列)
        # 座標に対してvisibleとhiddenがある
        # visibleには観測ノードの値をいれる
        # hiddenには隣接ノードから受け取るメッセージをいれる
        # メッセージは辞書型で、key:ノード座標, value:{0をとる確率、1をとる確率}
        for n in self.nodes():
            # 入力画像の値を観測値ノードのvalueにセットする
            # inputは画像の2次元表現を2次元配列で保持
            self.visible[n]['value'] = self.input[n[0], n[1]]

            # 潜在変数が隣接ノードから受け取るメッセージを辞書として保存
            # {ノード座標: 各値{0, 1 をとる確率}
            # f()でarray[1.0, 1.0]を返す。潜在変数について、index0は0をとる確率、index1は1をとる確率を表す
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

    @property
    def denoised(self):
        '''
        ノイズ除去した画像を生成
        '''
        # 確率伝播(Loopy Belief Propagation)
        for p in self.belief_propagation():
            pass

        denoised = np.copy(self.input)
        for r, c in self.nodes():
            prob = np.array([1.0, 1.0])
            # 隣接ノードからメッセージを受け取る
            messages = self.hidden[(r, c)]['messages']
            # 各隣接ノードからのメッセージを掛け合わせる(周辺分布)
            for value in messages.values():
                prob *= value
            # 周辺分布から潜在変数の推定値を算出
            denoised[r, c] = 0 if prob[0] > prob[1] else 1
        return denoised

    def send_message(self, source):
        '''
        sourceで指定したノードから各隣接ノードへメッセージを送信
        '''
        # メッセージを送信するターゲット
        # sourceの潜在変数
        # isinstanceはnがtupleのときtrueであり、tupleの場合のみ配列の要素とする
        targets = [n for n in self.hidden[source] if isinstance(n, tuple)]

        # 収束判定のために、前回のループ時のメッセージとの差分をとる
        diff = 0
        for target in targets:
            # source で指定されたノードの周辺分布を求める
            # 周辺化
            message = self.marginal(source, target)
            # 周辺分布
            message /= np.sum(message)

            # 前回ループ時のメッセージ
            messages = self.hidden[target]['messages']        

            # 前回ループ時のメッセージとの差分の絶対値を加算
            diff += np.sum(np.abs(messages[source] - message))
            
            # 今回メッセージを保存
            messages[source] = message

        # 差分の総和を返す
        return diff

    def marginal(self, source, target):
        '''
        sourceで指定したノードの周辺化
        '''
        m = np.array([0.0, 0.0])
        for i in range(2):
            # iの周辺分布を計算する
            prob = self.prob(i)
            # 隣接ノード
            neighbors = self.hidden[source]['messages'].keys()
            
            # targetノード以外の周辺分布を計算
            for n in [n for n in neighbors if n != target]:
                # targetノード以外のメッセージの積
                prob *= self.hidden[source]['messages'][n]

            # prob行列の要素の総和
            m[i] = np.sum(prob)
        return m


    def belief_propagation(self, loop=20):
        '''
        収束判定条件
        ここではグラフのエッジ数 * thresholdで指定された数値、変更可能
        '''
        # グラフのエッジを取得(潜在層のエッジ)
        edges = [e for e in self.hidden.edges()]
        # edgesの中からe[0]とe[1]がtupleのもののみがエッジ
        edges = [e for e in edges if isinstance(e[0], tuple) and isinstance(e[1], tuple)]
        
        threshold = self.threshold * len(edges)

        # self.nodesは、nodeの場所を表す行列のindexをもつtuple
        for n in self.nodes():
            # ノードnの周辺分布(prob)
            message = self.prob(self.visible[n]['value'])
            message /= np.sum(message)
            self.hidden[n]['messages'][n] = message
        yield

        for i in range(loop):
            diff = 0
            for n in self.nodes():
                diff += self.send_message(n)
                yield
            
            # 収束
            if diff < threshold:
                break

    def prob(self, value):
        '''
        周辺分布の計算
        '''
        # 0である確率: valueが0だったら1+self.theta、そうでなければ1-self.theta
        # 1である確率: valueが1だったら1+self.theta、そうでなければ1-self.theta
        base = np.array([1+self.theta if value == 0 else 1-self.theta, 1+self.theta if value == 1 else 1-self.theta])
        return base


def get_corrupted_input(img, corruption_level):
    '''
    画像へのノイズ付与
    '''
    # 元の画像をコピー
    corrupted = np.copy(img)

    # denoising autoencoderのときもこのノイズを付与した
    # corruption_levelで指定した割合だけ 1 になる(ノイズピクセルを表すフラグ)
    inv = np.random.binomial(n=1, p=corruption_level, size=img.shape)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            # (r, c)が1の場合
            if inv[r, c]:
                # ビット反転する(binaryなので0→1、1→0)
                corrupted[r, c] = ~(corrupted[r, c].astype(bool))
    return corrupted


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

    img = np.asarray([0,1,1,0,1,1,1,0])
    img = img.reshape(2,4)
    mrf = MRF(img)

def train(samplenum=5, dataset='data/mnist.pkl.gz'):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # 教師データのインプットとターゲット
    train_set_x, train_set_y = train_set
    
    # debug 表示
    pylab.figure()
    #for index in range(1):
    #    pylab.subplot(10, 10, index + 1)
    #    pylab.axis('off')
    #    pylab.imshow(train_set_x[index, :].reshape(28, 28), cmap=pylab.cm.gray, interpolation='nearest')
    #pylab.show()

    for i, index in enumerate(range(len(train_set_x))):
        if i >= samplenum:
            break
        img = train_set_x[index, :].reshape(28, 28)
        #print img
        img = (img >= 0.5).astype(int)
        #print img
        #debug
        #pylab.subplot(1, 1, 1)
        #pylab.axis('off')
        #pylab.imshow(img, cmap=pylab.cm.gray, interpolation='nearest')
        #pylab.show()

        # 5%のノイズを付与
        corrupted = get_corrupted_input(img, 0.05)

        # ノイズ付き画像からマルコフ確率場を生成
        mrf = MRF(corrupted)

        # 元の画像、ノイズ付き画像、ノイズ除去画像を表示
        pylab.subplot(1, 3, 1)
        pylab.axis('off')
        pylab.imshow(img, cmap=pylab.cm.gray)
    
        pylab.subplot(1, 3, 2)
        pylab.axis('off')
        pylab.imshow(mrf.input, cmap=pylab.cm.gray)

        pylab.subplot(1, 3, 3)
        pylab.axis('off')
        pylab.imshow(mrf.denoised, cmap=pylab.cm.gray)
        pylab.show()

if __name__ == '__main__':
    #test()

    train()
