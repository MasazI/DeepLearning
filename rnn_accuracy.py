import numpy
import pdb
import cPickle
import random
import os
import stat
import subprocess
from os.path import isfile, join
from os import chmod

def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words
    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename,'w')
    f.writelines(out)
    f.close()
    
    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'
    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    # ['accuracy:', '25.26%;', 'precision:', '0.08%;', 'recall:', '0.23%;', 'FB1:', '0.12']
    # print "out: " , out
    precision = float(out[3][:-3])
    recall    = float(out[5][:-3])
    f1score   = float(out[7])

    return {'p':precision, 'r':recall, 'f1':f1score}

def get_perfo(filename):
    ''' 
    work around for using a PERL script in python
    dirty but still works.
    '''
    tempfile = str(random.randint(1,numpy.iinfo('i').max)) + '.txt'
    chmod('conlleval.pl', stat.S_IRWXU) # give the execute permissions
    cmd = 'conlleval.pl < %s | grep accuracy > %s'%(filename,tempfile)
    print cmd
    out = os.system(cmd)
    out = open(tempfile).readlines()[0].split()
    os.system('rm %s'%tempfile)
    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])
    return {'p':precision, 'r':recall, 'f1':f1score}

if __name__ == '__main__':
    #print get_perf('valid.txt')
    print get_perf('valid.txt')
