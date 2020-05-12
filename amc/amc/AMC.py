import numpy as np
import _lda
import utils
import codecs
import re
import jieba
import math
from keras.preprocessing import sequence
from gensim.corpora import Dictionary
from sklearn import cross_validation
import logging

logger = logging.getLogger('amc')
#读取停止词
file = codecs.open('stopwords.dic','r','utf-8')
stoplist = [line.strip() for line in file] 
#读取数据集
file = codecs.open('data.dat','r','utf-8')
doc_set = [document.strip() for document in file]

texts = [] 
for i in doc_set: 
    raw = i.lower().strip()
    tokens = jieba.cut(raw)    
    stemmed_tokens = [word.strip() for word in tokens]
    stopped_tokens = [word for word in stemmed_tokens if word not in stoplist and len(word) > 1 and not re.search('[0-9]', word)]  
    texts.append(stopped_tokens)
dictionary = Dictionary(texts)
corpus =[dictionary.doc2idx(text) for text in texts]
corpus1=sequence.pad_sequences(corpus,maxlen=77)
trainset, testset= cross_validation.train_test_split(corpus1, test_size=0.2, random_state=0)
n_topics=10
random_state=0
n_iter=10

class AMC:   
    def __init__(self, n_topics, n_iter, alpha=0.1, eta=0.01, random_state=None,
                 refresh=10):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        rng = utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # configure console logging if not already configured
        

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
       
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
        self._fit(X)
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16):
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = utils.matrix_to_lists(X)
        # TODO: this loop is parallelizable
        for d in np.unique(DS):
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol)
        return doc_topic

    def _transform_single(self, doc, max_iter, tol):
        PZS = np.zeros((len(doc), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            PZS_new = self.components_[:, doc].T
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def _fit(self, X):
        random_state = utils.check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)
        for it in range(self.n_iter):
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("<{}> log likelihood: {:.0f}".format(it, ll))
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        ll = self.loglikelihood()
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self,self.components_,self.doc_topic_,self.topic_word_

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter
        

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        self.loglikelihoods_ = []

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return _lda._loglikelihood(nzw, ndz, nz, nd, alpha, eta)

    def _sample_topics(self, rands):
        """Samples all topic assignments. Called once per iteration."""
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        _lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                alpha, eta, rands)
    def perplexity(self):
        num_topics=10
        prep = 0.0
        prob_doc_sum = 0.0
        testset_word_num = 0
        topic_word_list=self.topic_word_
        doc_topics_list=self.doc_topic_
        for i in range(len(testset)):
            prob_doc = 0.0
            doc = testset[i]
            doc_word_num = 0 
            for word_id,word in enumerate (doc):
                prob_word = 0.0  
                doc_word_num +=1
                for topic_id in range(num_topics):
                    prob_topic = doc_topics_list[i][topic_id]
                    prob_topic_word = topic_word_list[topic_id][word_id]
                    prob_word += prob_topic * prob_topic_word
                prob_doc += math.log(prob_word)
            prob_doc_sum += prob_doc
            testset_word_num += doc_word_num
        prep = np.exp(-prob_doc_sum / testset_word_num)         
        print( "perplexity : %s" % prep) 

def KL():
    model = AMC(n_topics=10, random_state=0, n_iter=10)
    model.fit(trainset)
    D=[]
    phi=model.topic_word_
    for i in range(0,n_topics):
        for j in range(i+1,n_topics):
            d1=sum(phi[i]*np.log(2*phi[i]/(phi[i]+phi[j])))
            d2=sum(phi[j]*np.log(2*phi[j]/(phi[i]+phi[j])))
            d=(d1+d2)/2
            D.append(d)
    print("KL:", np.mean(D))

def run():
    model = AMC(n_topics, random_state)
    model.fit(trainset)
    model.perplexity()
    KL()

if __name__ == '__main__':
    run()