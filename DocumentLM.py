from collections import defaultdict
import logging
import os
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import gensim

logger = logging.getLogger(__name__)

class DocumentLM(object):
    
    
    def __init__(self, doc, dictionary, topic, topicAssignment, vocab):
        self.termFreq = defaultdict(int)    # Corpus frequency
        self.LM = {}
        #self.documentPath = documentPath
        self.vocab = vocab
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.topic = topic
        self.topicAssignment = topicAssignment
        self.doc = doc
        self.dictionary = dictionary
        
    def buildDocumentLM(self):
        tf = np.zeros(len(self.dictionary), dtype=np.float)  
        stoplist = set(nltk.corpus.stopwords.words("english"))  
        
        for tok in self.doc:
            if self.dictionary[tok[0]] not in stoplist:
                tf[tok[0]] += tok[1]
                
        docLen = tf[tf > 0].sum()
        #try:
        #    old_error_settings = np.seterr(divide='ignore')
        self.LM = self.div0(tf, docLen)
        #finally:
        #    np.seterr(**old_error_settings)
        self.termFreq = tf

    def buildTopicLM(self):
        tf = np.zeros(len(self.vocab), dtype=np.float)  
        for w in self.topic:
            #print w
            tf[self.vocab[w[1]]] += w[0]
                
        docLen = tf[tf > 0].sum()
        #try:
        #    old_error_settings = np.seterr(divide='ignore')
        self.LM = self.div0(tf, docLen)
        #finally:
        #    np.seterr(**old_error_settings)
        self.termFreq = tf

    def buildTARLM(self):
        tf = np.zeros(len(self.vocab), dtype=np.float)  
        for w in self.topicAssignment:
            tf[self.vocab[w]] += self.topicAssignment[w]
                
        docLen = tf[tf > 0].sum()
        #try:
        #    old_error_settings = np.seterr(divide='ignore')
        self.LM = self.div0(tf, docLen)
        #finally:
        #    np.seterr(**old_error_settings)
        self.termFreq = tf

    def div0(self, a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0  # -inf inf NaN
        return c

if __name__ == "__main__":
    modelDir = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(modelDir,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(modelDir,"mtsamples.mm"))
    DLM = DocumentLM(corpus[10], dictionary, "", "", "")
    DLM.buildDocumentLM()
    print DLM.LM
    print np.count_nonzero(DLM.LM)


        


