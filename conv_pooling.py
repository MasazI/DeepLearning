#coding: utf-8
## for convolutional
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

## for toy code
import pylab
from PIL import Image

# ランダムオブジェクトの生成
rng = np.random.RandomState(23455)

# 入力データのシンボルを作成
input = T.tensor4(name='input')

# 畳込みフィルターの初期化
# 2種類のフィルタ(出力の特徴マップ数)、3チャンネル(入力の特徴マップ数)、H=9、w=9
w_shp = (2,3,9,9)
# 画像のデータ次元のルート
w_bound = np.sqrt(3*9*9)
# 行列Wに初期値を与えての共有変数を定義（1/画像データ次元のルートのnegativeをmin、positiveをmaxとしてランダムに数値を生成、sizeはw_shpを指定するので、3*9*9の行列が2つできる）
W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shp), dtype=input.dtype), name='W')

# バイアス項の初期化
b_shp = (2,)
# -0.5から0.5の間で、b_shp(ここでは長さ2)の1次元ベクトルを生成
b = theano.shared(np.asarray(rng.uniform(low=-.5, high=.5, size=b_shp), dtype=input.dtype), name='b')

# 畳込み処理の定義
# inputデータにWをかける
conv_out = conv.conv2d(input, W)

# dimshuffleはTensorに対する演算を定義=演算をブロードキャストする次元を指定する
# ここではbiasを足し合わせるところに使われているので、2つのbiasは2つのフィルタそれぞれに対するもの
# と考えられる。conv_outの行列の次元と合うようにしているわけなので、conv_outは次元
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))

f = theano.function([input], output)

# toy code
img = Image.open(open('image/cat.jpg'))
width = img.size[0]
height = img.size[1]
img = np.asarray(img, dtype='float64') / 256.

# 画像の行列表現に変換
# np.transopse(img, (2,0,1))ともかける
# 2, 0, 1 は もとのimgのindex 2, 0, 1 が入れ替わってindex 0, 1, 2 に変更される。
# imgはもともとPILの画像オブジェクトなので、arrayにしても構造は変わらない
# [R,G,B]の配列が、画像の各ピクセルを表わすindexに配置される構造である
# 2, 0, 1と変更することによって、画像の各ピクセルを表わすindexに輝度情報を持つ行列を、R,G,Bの3つ保持する構造に変換する
# 上で変換した構造をreshapeで4次元Tensorにする
# reshapeはの画像1枚、チャネル3、height、widthは4DTensorへの変換である
# reshapeで追加している1軸目はdummy
img_ = img.transpose(2,0,1).reshape(1,3,height,width)

# 畳込み処理、結果は画素ごとに全チャンネルにわたって加算し、
# 出力はフィルタの数の特徴マップとなる
filtered_img = f(img_)

print "shape of original img: " + str(img.shape)
print "shape of original img 4dtensor: " + str(img_.shape)
print "shape of filter: " + str(w_shp)
# filterのクロップにより、上下左右4pxずつ減る
print "shape of filtered img 4dtensor: " + str(filtered_img.shape)

maxpool_shape = (2, 2)

input = T.dtensor4('input')

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f_maxpooling = theano.function([input], pool_out)
pool_img = f_maxpooling(filtered_img)

print "shape of maxpool img: " + str(pool_img.shape)

# 結果
# pylabは画像をプロットする利用
# original image
pylab.subplot(2,3,1)
#pylab.axis('off')
pylab.imshow(img)
pylab.gray()

# 畳込み画像1
pylab.subplot(2,3,2)

# 2つのフィルタで畳み込んでいるので2種類の画像ができる
# filtereはminibatchなので、最初のindexは0指定する。
pylab.imshow(filtered_img[0,0,:,:])

# 畳画像2
pylab.subplot(2,3,3)
pylab.imshow(filtered_img[0,1,:,:])

# maxpooling1
pylab.subplot(2,3,4)
pylab.imshow(pool_img[0,0,:,:])

# maxpooling2
pylab.subplot(2,3,5)
pylab.imshow(pool_img[0,1,:,:])

# 表示
pylab.savefig('log/convolution_pooling_sample.png')
pylab.show()


