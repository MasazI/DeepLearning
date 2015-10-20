#coding: utf-8

import numpy as np
import timeit
import sys
import os
import subprocess
import random
import copy

import rnn_load
import rnn_tools
import rnn_elman
import rnn_accuracy

# python の最大再帰数の設定（Cスタックがオーバーフローしてクラッシュすることを防ぐ）
sys.setrecursionlimit(1500)

def main(param=None):
    if not param:
        param = {
            'data': 'atis',
            'lr': 0.0970806646812754,
            'verbose': 1,
            'decay': True, # wait decay
            'win': 7, # number of words in the context window
            'nhidden': 200, # the number of hidden layer's units
            'seed': 345,
            'emb_dimension': 50, # the dimension of word embedding
            'nepochs': 60, # the number of epochs
            'savemodel': True # savie model flag
        }

    print param

    log_folder_name = './log'
    model_folder_name = './model/rnn'
    log_folder = os.path.join(os.path.dirname(__file__), log_folder_name)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    model_folder = os.path.join(os.path.dirname(__file__), model_folder_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    train_set, test_set, dic = rnn_load.atis_all()

    # dicからindexとラベル, indexとワードの組を生成
    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

    # setを分割しておく
    train_lex, train_ne, train_y = train_set
    
    # test_setからvalidationを生成
    val_endindex =  len(test_set[0])/2 #lex, ne, y全て同じ要素数
    test_endindex = len(test_set[0]) #lex, ne, y全て同じ要素数

    valid_set = [test_set[0][0 : val_endindex], test_set[1][0 : val_endindex], test_set[2][0 : val_endindex]]
    test_set = [test_set[0][val_endindex : test_endindex], test_set[0][val_endindex : test_endindex], test_set[2][val_endindex : test_endindex]]
    
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set 

    # ボキャブラリーサイズ
    vocsize = len(set(reduce(lambda x, y: list(x) + list(y), train_lex + valid_lex + test_lex)))

    # クラス数
    nclasses = len(set(reduce(lambda x, y: list(x) + list(y), train_y + valid_y + test_y)))

    # センテンス数
    nsentences = len(train_lex)

    # 出力
    training_data = {'vocsize': vocsize, 'nclasses': nclasses, 'nsentences': nsentences}

    # validation正解(groundtruth)データ
    # ラベル(ラベルをインデックス)
    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    # 単語
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    
    # test正解(groundtruth)データ
    # ラベル
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    # 単語
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    # instanciate the model
    np.random.seed(param['seed'])
    random.seed(param['seed'])

    rnn = rnn_elman.RNNSLU(nh=param['nhidden'], nc=nclasses, ne=vocsize, de=param['emb_dimension'], cs=param['win'])

    # train
    best_f1 = -np.inf
    param['clr'] = param['lr']
    for e in xrange(param['nepochs']):
        # network に与えるデータをシャッフル
        rnn_tools.shuffle([train_lex, train_ne, train_y], param['seed'])
        
        # epoch数のcount
        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            # rnnの学習
            # xにはsentenceのwords、yにはそのラベルが入る. ここではsentenceごとに学習する
            rnn.train(x, y, param['win'], param['clr'])

            print '[leaning] epoch %i >> %2.2f %%' % (e, (i+1) * 100. / nsentences),
            print 'completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic),
            sys.stdout.flush()

            # testのボキャブラリーでループし、正解ラベルと分類結果のMapを作成
            predictions_test = [map(lambda x: idx2label[x], rnn.classify(np.asarray(rnn_tools.contextwin(x, param['win'])).astype('int32'))) for x in test_lex]
            
            # validationのボキャブラリーでループし、正解ラベルと分類結果のMapを作成
            predictions_valid = [map(lambda x: idx2label[x], rnn.classify(np.asarray(rnn_tools.contextwin(x, param['win'])).astype('int32'))) for x in valid_lex]

            # evaluation using colleval (NLPもっと勉強しないとここ理解難しい)
            res_test  = rnn_accuracy.conlleval(predictions_test, groundtruth_test, words_test, log_folder + '/current.test.txt')
            res_valid = rnn_accuracy.conlleval(predictions_valid, groundtruth_valid, words_valid, log_folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            if param['savemodel']:
                rnn.save(model_folder)

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid['f1']

            print('NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = e

            # logをbestに移す
            subprocess.call(['mv', log_folder + '/current.test.txt', log_folder + '/best.test.txt'])
            subprocess.call(['mv', log_folder + '/current.valid.txt', log_folder + '/best.valid.txt'])
        else:
            print ''

        if param['decay'] and abs(param['be'] - param['ce']) >= 10:
            # ステップごとに学習係数を小さくしていく
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break
            
    print('BEST REULST: epoch', param['be'], 'valid F1', param['vf1'], 'best test F1', param['tf1'],'with the nodel', folder)

if __name__ == '__main__':
    main()


