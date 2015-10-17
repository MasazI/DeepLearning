#coding: utf-8
import numpy as np
import pdb
import cPickle
import urllib
import os
import sys
from os.path import isfile

# データのprefixを作成しておく
PREFIX = os.getenv('ATISDATA', os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0], 'ML/data'))

# atisデータのロード
def atis_all():
    filename = os.path.join(PREFIX, 'atis.pkl')
    train_set, test_set, dicts = cPickle.load(open(filename))
    return train_set, test_set, dicts

def load_data_test():
    # atisデータ
    data = atis_all()
    train, test, dic = data
    
    # 辞書データを展開する配列
    w2ne, w2la = {}, {}

    # wordからindex、テーブルからindex, labelからindex
    w2idx, ne2idx, label2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    # 逆の辞書も作っておく
    idx2w = dict((v,k) for k,v in w2idx.iteritems())
    idx2ne = dict((v,k) for k,v in ne2idx.iteritems())
    idx2la = dict((v,k) for k,v in label2idx.iteritems())

    # データをカラムごとに保持
    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train

    # 見やすく表示するためのwidth
    wlength = 50
    
    print "test1. index to word"
    sentence = np.array([383, 189, 13, 193, 208, 307, 195, 502, 260, 539, 7, 60, 72, 8, 350, 384], dtype=np.int32)
    print sentence, str(len(sentence))
    print map(lambda x: idx2w[x], sentence)
   
    print "test2. index to label"
    labels = np.array([126, 126, 126, 126, 126, 48, 50, 126, 48, 50, 126, 78, 123, 81, 126, 15, 14, 89, 89], dtype=np.int32)
    print labels, str(len(labels))
    print map(lambda x: idx2la[x], labels)


    print "test3. map of atis data. you input c to forward iterate."
    for e in ['train', 'test']:
        for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
            print 'WORD'.rjust(wlength), 'LABEL'.rjust(wlength)
            for wx, la in zip(sw, sl):
                print idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength)
            print '\n' + '**'*30 + '\n'
            pdb.set_trace()
            
if __name__ == '__main__':
    load_data_test()
