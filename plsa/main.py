from preprocess import Preprocess as PP
from plsa import PLSA
import numpy as np
import logging
import time
def main():
    # setup logging --------------------------
    logging.basicConfig(filename='plsa.log',
                        level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    #console = logging.StreamHandler()
    #console.setLevel(logging.INFO)
    #logging.getLogger('').addHandler(console)
    # some basic configuration ---------------
    fname = './data.txt'
    fsw = './stopwords.txt'
    eps = 20.0
    key_word_size = 10
    # preprocess -----------------------------
    pp = PP(fname, fsw)
    w_d = pp.get_w_d()
    V, D = w_d.shape
    logging.info('V = %d, D = %d' % (V, D))
    # train model and get result -------------
    pmodel = PLSA()
    for z in range(3, (D+1), 10):
        t1 = time.clock()
        (l, p_d_z, p_w_z, p_z) = pmodel.train(w_d, z, eps)
        t2 = time.clock()
        #logging.info('z = %d, eps = %f, time = %f' % (z, l, t2-t1))
        #print ('z = %d, eps = %f, time = %f' % (z, l, t2-t1))
        for itz in range(z):
            logging.info('Topic %d' % itz)
            data = [(p_w_z[i][itz], i) for i in range(len(p_w_z[:,itz]))]
            data.sort(key=lambda tup:tup[0], reverse=True)
            for i in range(key_word_size):
                logging.info('%s : %.6f ' % (pp.get_word(data[i][1]), data[i][0]))
                print ('%s : %.6f ' % (pp.get_word(data[i][1]), data[i][0]))
def perplexity():
    phi = p_w_z
    log_per = 0
    N = 0
    for m,doc in enumerate(w_d.T):
        theta = p_d_z[m]/(sum(doc)+z*0.1)
        for i,count in enumerate (w_d[:,m]):
            if count>0:        
                log_per -= numpy.log(numpy.inner(phi[i,:], theta))
        N += sum(doc)
    return (np.exp(log_per / N))

a=p_w_z
def KL():
    for i in range(0,K):
        if i<K-1:
            d1=sum(a[i]*np.log(2*a[i]/(a[i]+a[i+1])))
            d2=sum(a[i+1]*np.log(2*a[i+1]/(a[i]+a[i+1])))
            return((d1+d2)/2) 