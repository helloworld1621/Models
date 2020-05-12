from gensim.models import LdaModel
from gensim.corpora import Dictionary
import jieba
import codecs
import re
import numpy as np
import math
from sklearn import cross_validation

#读取停止词
file = codecs.open('stopwords.dic','r','utf-8')
stoplist = [line.strip() for line in file] 
#读取数据集
file = codecs.open('data.txt','r','utf-8')
doc_set = [document.strip() for document in file]
   
texts = [] 
for i in doc_set: 
    raw = i.lower().strip()
    tokens = jieba.cut(raw)    
    stemmed_tokens = [word.strip() for word in tokens]
    stopped_tokens = [word for word in stemmed_tokens if word not in stoplist and len(word) > 1 and not re.search('[0-9]', word)]  
    texts.append(stopped_tokens)
    
num_topics=10
num_words=10
    
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
trainset, testset= cross_validation.train_test_split(corpus, test_size=0.2, random_state=0)
lda = LdaModel(corpus=trainset, id2word=dictionary, num_topics=num_topics)
lda.print_topics(num_topics=num_topics, num_words=num_words)

def perplexity():
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] 
    for topic_id in range(num_topics):
        topic_word = lda.show_topic(topic_id, len(dictionary))
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)  
    doc_topics_ist = []  
    for doc in testset:
        doc_topics_ist.append(lda.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0  
        for word_id, num in dict(doc).items():
            prob_word = 0.0  
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print( "模型困惑度的值为 : %s" % prep)

def KL():
    phi=np.zeros((num_topics,len(dictionary)),dtype="float64")
    for topic_id in range(num_topics):
        topic_word = lda.show_topic(topic_id, len(dictionary))
        s=[]
        for word, probability in topic_word:
            s.append(probability)
        phi[topic_id]+=s 
        
    D=[]
    for i in range(0,num_topics):
        for j in range(i+1,num_topics):
            d1=sum(phi[i]*np.log(2*phi[i]/(phi[i]+phi[j])))
            d2=sum(phi[j]*np.log(2*phi[j]/(phi[i]+phi[j])))
            d=(d1+d2)/2
            D.append(d)
    print( np.mean(D))


def run():
    perplexity()
    KL()

if __name__ == '__main__':
    run()
    
