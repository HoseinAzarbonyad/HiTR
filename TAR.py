from CollectionLM import CollectionLM
from DocumentLM import DocumentLM
from ParsimoniousLM import ParsimoniousLM
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

class TAR(object):
    
    
    def __init__(self, corpus, modelPath, outputPath, mu, threshold, numIteration):
        self.termFreq = defaultdict(int)    # Corpus frequency
        self.LM = {}
        #self.documentsPath = documentsPath
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.vocab = vocab = {} 
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.outputPath = outputPath
        self.vocab = vocab = {} 
        self.modelPath = modelPath
        self.mu = mu
        self.threshold = threshold
        self.numIteration = numIteration
        self.corpus = corpus


    def _assignTopics(self):       
        model = gensim.models.LdaModel.load(os.path.join(self.modelPath, "lda.model"))
        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.modelPath,"mtsamples.dict"))

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
 
        stoplist = set(nltk.corpus.stopwords.words("english"))  
        for f in os.listdir(self.documentsPath):
            #print f
            currDir = os.path.join(self.documentsPath, f)
            if os.path.isdir(currDir):
                #print f
                if not os.path.exists(os.path.join(self.outputPath, f)):
                    os.makedirs(os.path.join(self.outputPath, f))
                for f2 in os.listdir(os.path.join(self.documentsPath,f)):
                    fin = open(os.path.join(self.documentsPath, f, f2), 'rb')
                    text = fin.read()
                    #print doc
                    fin.close()
                    doc = self.tokenizer.tokenize(text)
                    doc = dictionary.doc2bow(doc)
                    #bow = MyCorpus(os.path.join(TEXTS_DIR, f))
                    x = model[doc]
                    outfile = open(os.path.join(self.outputPath, f, f2), "w")
                    for t in x:
                        outfile.write(str(t))
                        outfile.write("\n")


    def assignTopics(self):       
        model = gensim.models.LdaModel.load(os.path.join(self.modelPath, "lda-TR.model"))
        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.modelPath,"mtsamples.dict"))
        topic_assignments = []
        for doc in self.corpus:
            x = model[doc]
            topic_assignments.append(x)
        return topic_assignments

    def runTAR(self):
        topic_assignments = self.assignTopics()
        #CLM = CollectionLM("", "", self.outputPath)
        CLM = CollectionLM("", "", "", topic_assignments)
        CLM.TARCollectionLM()
        parsimonized_topic_assignments = []
        for doc in topic_assignments:
            docTopicAssignment = {}
            for t in doc:
                topic = t[0];
                prob = t[1];
                docTopicAssignment[topic] = prob
            DLM = DocumentLM("", "", "", docTopicAssignment, CLM.vocab)
            DLM.buildTARLM();
            #PLM = ParsimoniousLM(DLM.LM, CLM.vocab, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
            PLM = ParsimoniousLM(CLM.vocab, DLM.LM, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
            PLM.parsimonize()
            doc_topics = []
            for tok in PLM.vocab.keys():
                if PLM.docLM[PLM.vocab[tok]] > 0:
                    tup = (tok, PLM.docLM[PLM.vocab[tok]])
                    doc_topics.append(tup)
            parsimonized_topic_assignments.append(doc_topics)
        return parsimonized_topic_assignments

    def docTopics(doc):
        CLM = CollectionLM("", "", topic_assignments)
        CLM.TARCollectionLM()
        docTopicAssignment = {}
        for t in doc:
            topic = t[0];
            prob = t[1];
            docTopicAssignment[topic] = prob
        DLM = DocumentLM("", "", docTopicAssignment, CLM.vocab)
        DLM.buildTARLM();
        PLM = ParsimoniousLM(DLM.LM, CLM.vocab, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
        PLM.parsimonize()
        doc_topics = []
        for tok in PLM.vocab.keys():
            if PLM.docLM[PLM.vocab[tok]] > 0:
                tup = (tok, PLM.docLM[PLM.vocab[tok]])
                doc_topics.append(tup)
        return doc_topics


if __name__ == "__main__":

    ldaPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    documentsPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened"
    outPutPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-TAR"
    tar = TAR(ldaPath, documentsPath, outPutPath, 0.5, 0.001, 10)
    tar.runTAR()    
