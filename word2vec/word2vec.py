import os
import re
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import jieba
import codecs
from sklearn import cross_validation

#读取停止词
file = codecs.open('../data/stopwords.dic','r','utf-8')
stoplist = [line.strip() for line in file] 
#读取数据集
file = codecs.open('../data/data.txt','r','utf-8')
doc_set = [document.strip() for document in file]

doc=[]
for i in doc_set: 
    raw = i.lower().strip()
    tokens = jieba.cut(raw)    
    stemmed_tokens = [word.strip() for word in tokens]
    stopped_tokens = [word for word in stemmed_tokens if word not in stoplist and len(word) > 1 and not re.search('[0-9]', word)]  
    doc.append(stopped_tokens)

trainset, testset= cross_validation.train_test_split(doc, test_size=0.2, random_state=0)
num_features=100
min_word_count=4
num_workers=1
context=10
num_iter=1000
downsampling = 1e-3
K=10
alpha=50/K


class word2vec(object):
    def __init__(self,trainset,testset,num_features,min_word_count,num_workers,context,num_iter,K,alpha):
    # 设定训练的参数
        self.trainset=trainset
        self.testset=testset
        self.num_features=num_features    # Word vector dimensionality
        self.min_word_count=min_word_count  # Minimum word count 
        self.num_workers=num_workers      # Number of threads to run in parallel
        self.context=context           # Context window size
        self.num_iter=num_iter
        self.downsampling = downsampling   # Downsample setting for frequent words
        self.K=K
        self.alpha=alpha
    def model(self):
        model_name = '{}features_{}minwords_{}context.model'.format(self.num_features, self.min_word_count, self.context)

        self.model = Word2Vec(self.trainset, workers=self.num_workers, \
            size=self.num_features, min_count = self.min_word_count, \
            window = self.context,sample = self.downsampling)
        self.model.init_sims(replace=True)
        self.model.save(model_name)
    def cluster(self):
        word_vectors=[]
        words=[]
        for document in self.testset:
            for word in document:
                if word in self.model.wv.index2word:
                    words.append(word)
                    word_vectors.append(self.model[word])
        #word_vectors =self.model.syn1neg
        num_clusters=self.K
         
        kmeans_clustering = KMeans(n_clusters = num_clusters, n_jobs=4)
        idx = kmeans_clustering.fit_predict(word_vectors)

        x=kmeans_clustering.transform(word_vectors)
        self.values=[]
        for i in range(len(x)):
            for index,value in enumerate(x[i]):
                if value==min(x[i]):
                    self.values.append(value)

        self.word_centroid_map = dict(zip(words, idx))
        self.word_centroid_map2 = dict(zip(self.values, idx))
        self.word_centroid_map3 = dict(zip(words,self.values))

        for cluster in range(0,self.K):  
            y=len ([w for w,c in self.word_centroid_map.items() if c==cluster])
            if y>=10:
                print("\nCluster %d" % cluster)
                print([w for w,c in self.word_centroid_map.items() if c == cluster])
                print ([round(w,4) for w,c in self.word_centroid_map2.items() if c==cluster])

    def normalize(self,z):
        return (z - np.min(z))/(np.max(z) - np.min(z))

    def perplexity(self):
        S=[]
        log_per = 0
        Kalpha = self.K * self.alpha
        W=[]
        for m in range(len(self.testset)):
            for w in self.testset[m]:
                if w in self.word_centroid_map3:
                    s=np.log(self.word_centroid_map3[w]+Kalpha)/np.log(0.5)
                    W.append(str(w))
                    S.append(s)
        for i in self.normalize(S):
            # 将i的取值范围转化为word_centroid_map3原来的范围
            x=[v for k, v in self.word_centroid_map3.items() if v!=0]
            slope=(max(self.normalize(S))-max(self.word_centroid_map3.values()))/min(x)
            i=(max(self.normalize(S))-i)/slope
            log_per-=np.log(i+1e-3)
        print (np.exp(log_per / len(W)))

    def KL(self):
        phii=[]
        for i in range(0,self.K):
            x=len ([w for w,c in self.word_centroid_map.items() if c==i])
            if x>=10:
                phi=[round(w,4) for w,c in self.word_centroid_map2.items() if c==i]
                p=list (np.log(phi)/np.log(0.5))
                p.sort(reverse=True)
                phii.append(p[0:10]) 
        phii=np.array(phii)    

        D=[]
        for i in range(0,len(phii)):
            for j in range (i+1,len(phii)):
                d1=sum(phii[i]*np.log(2*phii[i]/(phii[i]+phii[j])))
                d2=sum(phii[j]*np.log(2*phii[j]/(phii[i]+phii[j])))
                d=(d1+d2)/2
                D.append(d)
        print (np.mean(D))

def run():
    Model=word2vec(trainset,testset,num_features,min_word_count,num_workers,context,K,alpha)
    Model.model()
    Model.cluster()
    Model.perplexity()
    Model.KL()

if __name__ == '__main__':
    run()