#-*- coding:utf-8 -*-
import logging
import logging.config
import configparser
import numpy as np
import random
import codecs
import os

import jieba
import re
from collections import OrderedDict
from sklearn import cross_validation
#获取当前路径
path = os.getcwd()
#导入日志配置文件
##logging.config.fileConfig("logging.conf") 
#创建日志对象
logger = logging.getLogger()
# loggerInfo = logging.getLogger("TimeInfoLogger")
# Consolelogger = logging.getLogger("ConsoleLogger")

#导入配置文件
conf = configparser.ConfigParser()
conf.read("setting.conf",encoding='utf-8') 
#文件路径
trainfile = os.path.join(path,os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","wordidmapfile")))
thetafile = os.path.join(path,os.path.normpath(conf.get("filepath","thetafile")))
phifile = os.path.join(path,os.path.normpath(conf.get("filepath","phifile")))
paramfile = os.path.join(path,os.path.normpath(conf.get("filepath","paramfile")))
topNfile = os.path.join(path,os.path.normpath(conf.get("filepath","topNfile")))
tassginfile = os.path.join(path,os.path.normpath(conf.get("filepath","tassginfile")))
#模型初始参数
K = 1
alpha = 50/K
beta = float(conf.get("model_args","beta"))
iter_times = int(conf.get("model_args","iter_times"))
top_words_num = int(conf.get("model_args","top_words_num"))

file = codecs.open('data/stopwords.dic','r','utf-8')
stoplist = [line.strip() for line in file] 
#读取数据集
file = codecs.open('data/data.txt','r','utf-8')
doc_set = [document.strip() for document in file]

file=codecs.open('data/noise_word.dat','r','utf-8')
ndoc_set=[word.strip() for word in file]

f=codecs.open('data/text.dat','w','utf-8')
for i in doc_set:
    raw = i.lower().strip()
    tokens = jieba.cut(raw)
    for word in tokens:       
        if word not in stoplist and not re.search('[0-9]',word):
            stemmed_tokens = word.strip()            
            f.write(str(stemmed_tokens)+' ')
    f.write('\n')   
f.close()

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class DataPreProcessing(object):

    def __init__(self):
        #self.docs_count = 0
        self.train_docs_count=0
        self.test_docs_count=0
        #self.words_count = 0
        self.train_words_count=0
        self.test_words_count=0
        #self.docs = []
        self.train_docs=[]
        self.test_docs=[]
        self.word2id = OrderedDict()
        self.words_n=[]

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w','utf-8') as f:
            for word,id in self.word2id.items():
                f.write(word +"\t"+str(id)+"\n")

class NRMModel(object):
    
    def __init__(self,dpre):

        self.dpre = dpre #获取预处理参数

        #
        #模型参数
        #聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        #
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num 
        #
        #文件变量
        #分好词的文件trainfile
        #词对应id文件wordidmapfile
        #文章-主题分布文件thetafile
        #词-主题分布文件phifile
        #每个主题topN词文件topNfile
        #最后分派结果文件tassginfile
        #模型训练选择的参数文件paramfile
        #
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile
        # p,概率向量 double类型，存储采样的临时变量
        # nw,词word在主题topic上的分布
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的词的总数
        # ndsum,每各doc中词的总数
        self.p = np.zeros(self.K)        
        self.nw = np.zeros((self.dpre.train_words_count,self.K),dtype="int")      
        self.nwsum = np.zeros(self.K,dtype="int")    
        self.nd = np.zeros((self.dpre.train_docs_count,self.K),dtype="int")       
        self.ndsum = np.zeros(dpre.train_docs_count,dtype="int")    
        self.Z = np.array([ [0 for y in range(dpre.train_docs[x].length)] for x in range(dpre.train_docs_count)]) 





        #随机先分配类型
        for x in range(len(self.Z)):
            self.ndsum[x] = self.dpre.train_docs[x].length
            for y in range(self.dpre.train_docs[x].length):
                topic = random.randint(0,self.K-1)
                self.Z[x][y] = topic
                self.nw[self.dpre.train_docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1

        self.theta = np.array([ [0.0 for y in range(self.K)] for x in range(self.dpre.train_docs_count) ])
        self.phi = np.array([ [ 0.0 for y in range(self.dpre.train_words_count) ] for x in range(self.K)]) 
    def sampling(self,i,j):

        topic = self.Z[i][j]
        word = self.dpre.train_docs[i].words[j]
        self.nw[word][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1

        #Vbeta = self.dpre.words_count * self.beta
        V1=self.dpre.train_words_count-len(self.dpre.words_n)
        V2=len(self.dpre.words_n)
        beta1=self.beta
        beta2=self.beta
        Kalpha = self.K * self.alpha

        if word not in self.dpre.words_n:
            self.p = (self.nw[word] + beta1)/(self.nwsum + V1*beta1) * \
                 (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)  
        else:
            self.p=(self.nw[word]+beta2)/(self.nwsum+V2*beta2)*\
            (self.nw[word]+beta2)/sum(self.nw[word]+V2*beta2)*\
            (self.nd[i]+self.alpha)/(self.ndsum[i]+Kalpha) 

        for k in range(1,self.K):
            self.p[k] += self.p[k-1]

        u = random.uniform(0,self.p[self.K-1])
        for topic in range(self.K):
            if self.p[topic]>u:
                break

        self.nw[word][topic] +=1
        self.nwsum[topic] +=1
        self.nd[i][topic] +=1
        self.ndsum[i] +=1
        return topic

    def est(self):
        # Consolelogger.info(u"迭代次数为%s 次" % self.iter_times)
        for x in range(self.iter_times):
            for i in range(self.dpre.train_docs_count):
                for j in range(self.dpre.train_docs[i].length):
                    topic = self.sampling(i,j)
                    self.Z[i][j] = topic
        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug(u"保存模型")
        self.save()
        
    def _theta(self):
        for i in range(self.dpre.train_docs_count):
            self.theta[i] = (self.nd[i]+self.alpha)/(self.ndsum[i]+self.K * self.alpha)
    def _phi(self):
        for i in range(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta)/(self.nwsum[i]+self.dpre.train_words_count * self.beta)
            
    def perplexity(self):
        #if docs == None: docs = docs
        #phi = phi
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        #for m, doc in enumerate(docs):
        for m in range(self.dpre.test_docs_count):
            # theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            theta = (self.nd[m]+self.alpha)/(self.ndsum[m]+Kalpha)
            for w in self.dpre.test_docs[m].words:
                log_per -= np.log(np.inner(self.phi[:,w],theta))
                #print('log_per1',log_per)
            N += self.dpre.test_docs[m].length
        print("perplexity:", np.exp(log_per / N))



    def KL(self):
        D=[]
        for i in range(0,self.K):
            for j in range(i+1,self.K):
                d1=sum(self.phi[i]*np.log(2*self.phi[i]/(self.phi[i]+self.phi[j])))
                d2=sum(self.phi[j]*np.log(2*self.phi[j]/(self.phi[i]+self.phi[j])))
                d=(d1+d2)/2
                D.append(d)
        print("KL:", np.mean(D))


    def save(self):
        #保存theta文章-主题分布
        logger.info(u"文章-主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile,'w') as f:
            for x in range(self.dpre.train_docs_count):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        #保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile,'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.train_words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        #保存参数设置
        logger.info(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile,'w','utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
        #保存每个主题topic的词
        logger.info(u"主题topN词已保存到%s" % self.topNfile)

        with codecs.open(self.topNfile,'w','utf-8') as f:
            self.top_words_num = min(self.top_words_num,self.dpre.train_words_count)
            for x in range(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = []
                twords = [(n,self.phi[x][n]) for n in range(self.dpre.train_words_count)]
                twords.sort(key = lambda i:i[1], reverse= True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value:key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')
        #保存最后退出时，文章的词分派的主题的结果
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)
        with codecs.open(self.tassginfile,'w') as f:
            for x in range(self.dpre.train_docs_count):
                for y in range(self.dpre.train_docs[x].length):
                    f.write(str(self.dpre.train_docs[x].words[y])+':'+str(self.Z[x][y])+ '\t')
                f.write('\n')
        logger.info(u"模型训练完成。")






def preprocessing():
    logger.info(u'载入数据......')
    with codecs.open(trainfile, 'r','utf-8') as f:
        docs = f.readlines()
    trainset, testset= cross_validation.train_test_split(docs, test_size=0.2, random_state=0)    
    
    logger.debug(u"载入完成,准备生成字典对象和统计文本数据...")
    dpre = DataPreProcessing()
    items_idx = 0
    for line in trainset:
        if line != "":
            tmp = line.strip().split()
            #生成一个文档对象
            doc = Document()
            for item in tmp:
                if item in dpre.word2id:
                    doc.words.append(dpre.word2id[item])
                else:
                    dpre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dpre.train_docs.append(doc)
        else:
            pass
        
    test_words_count = []   
    for line in testset:
        if line != "":
            tmp = line.strip().split()
            #生成一个文档对象
            test_doc = Document()
            word=[]
            for item in tmp:               
                if item in dpre.word2id:    
                    test_words_count.append(item)
                    word.append(item)
                    test_doc.words.append(dpre.word2id[item])                   
            test_doc.length = len(word)
            dpre.test_docs.append(test_doc)
        else:
            pass
        
    for word in ndoc_set:
        if word in dpre.word2id:
            dpre.words_n.append (dpre.word2id[word])
        else:
            pass
        
    dpre.train_docs_count = len(dpre.train_docs)
    dpre.test_docs_count = len(dpre.test_docs)
    dpre.train_words_count = len(dpre.word2id)
    dpre.test_words_count = len(set(test_words_count))
    logger.info(u"共有%s个文档" % dpre.train_docs_count)
    dpre.cachewordidmap()
    logger.info(u"词与序号对应关系已保存到%s" % wordidmapfile)
    return dpre   

def run():
    dpre = preprocessing()
    lda = NRMModel(dpre)
    lda.est()
    lda.perplexity()
    lda.KL()
    

if __name__ == '__main__':
    run()
    