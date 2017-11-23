from collections import defaultdict
from heapq import nlargest
import logging
import os
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import gensim


logger = logging.getLogger(__name__)

class CollectionLM(object):
    
    
    def __init__(self, corpus, dictionary, topics, topic_assignments):
        #logger.info("Building collection's language model")
        
        self.termFreq = defaultdict(int)    # Corpus frequency
        self.LM = {}
        self.corpus = corpus
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.dictionary = dictionary 
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.topics = topics
        #self.topicAssignmentsPath = topicAssignmentsPath
        self.topic_assignments = topic_assignments
        self.vocab = vocab = {}
    
    def buildCorpusLM(self):    
        count = defaultdict(int)
        stoplist = set(nltk.corpus.stopwords.words("english"))  
        for doc in self.corpus:
            for tok in doc:
                if self.dictionary[tok[0]] not in stoplist:
                    count[tok[0]] += tok[1]
        cf = np.empty(len(self.dictionary), dtype=np.float)
        
        for i, f in count.iteritems():
            cf[i] = f
        
        try:
            old_error_settings = np.seterr(divide='ignore')

            self.LM = cf / np.sum(cf)
        finally:
            np.seterr(**old_error_settings)
        
        self.termFreq = count
    def buildTMcollectionLM(self):
        count = defaultdict(float)
        
        for topic in self.topics:
            for w in self.topics[topic]:
                i = self.vocab.setdefault(w[1], len(self.vocab))
                count[i] += w[0]
        
        cf = np.empty(len(count), dtype=np.float)
        
        for i, f in count.iteritems():
            cf[i] = f
        
        try:
            old_error_settings = np.seterr(divide='ignore')

            self.LM = cf / np.sum(cf)
        finally:
            np.seterr(**old_error_settings)


    def _TARCollectionLM(self):
        count = defaultdict(int)
        for f in os.listdir(self.topicAssignmentsPath):
            #print f
            currDir = os.path.join(self.topicAssignmentsPath, f)
            if os.path.isdir(currDir):
                #print f
                for f2 in os.listdir(os.path.join(self.topicAssignmentsPath,f)):
                    fin = open(os.path.join(self.topicAssignmentsPath, f, f2), 'rb')
                    for line in fin.readlines():
                        topic = line.split(" ")[0].split("(")[1].split(",")[0];
                        prob = float(line.split(" ")[1].split(")")[0]);
                        i = self.vocab.setdefault(topic, len(self.vocab))
                        count[i] += prob

        cf = np.empty(len(count), dtype=np.float)
        
        for i, f in count.iteritems():
            cf[i] = f
        
        try:
            old_error_settings = np.seterr(divide='ignore')

            self.LM = cf / np.sum(cf)
        finally:
            np.seterr(**old_error_settings)

    def TARCollectionLM(self):
        count = defaultdict(int)
        for doc in self.topic_assignments:
            for t in doc:
                topic = t[0]
                prob = t[1]
                i = self.vocab.setdefault(topic, len(self.vocab))
                count[i] += prob

        cf = np.empty(len(count), dtype=np.float)
        
        for i, f in count.iteritems():
            cf[i] = f
        
        try:
            old_error_settings = np.seterr(divide='ignore')

            self.LM = cf / np.sum(cf)
        finally:
            np.seterr(**old_error_settings)
  

if __name__ == "__main__":
    modelDir = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(modelDir,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(modelDir,"mtsamples.mm"))
    CLM = CollectionLM(corpus, dictionary, "", "")
    CLM.buildCorpusLM()
    print CLM.LM



        

