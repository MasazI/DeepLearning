# coding: utf-8
import random
import numpy as np

def contextwin(l, win):
    """ win:: int windowsのサイズと一致する必要がある
        l:: array 単語のindexの入った配列
        return:: 文章の各ワードを含むコンテキストウィンドウと一致するインデックスの配列を配列でかえす   
                 はみ出る分についてはindex -1がかえる
    """
    # windowはuneven
    assert(win%2) == 1
    # windowは1以上
    assert win >= 1
    # python標準のlist型にする
    l = list(l)

    # 端のワードはwindowがはみ出るのでindex -1 を足しておく(// は小数点以下切り捨ての除算)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]

    # lpaddedには
    out = [ lpadded[i : (i+win)] for i in range(len(l)) ]

    # outの長さは元の配列の長さと同じはず（要素が各ワードに対するコンテキストウィンドウに変わるだけ）
    assert len(out) == len(l)
    return out

def contextwin_test():
    sentence = np.array([383, 189, 13, 193, 208, 307, 195, 502, 260, 539, 7, 60, 72, 8, 350, 384], dtype=np.int32)
    wins = contextwin(sentence, 5)

    print wins

if __name__ == '__main__':
    contextwin_test()
