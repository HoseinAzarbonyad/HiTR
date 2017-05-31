from CollectionLM import CollectionLM
from DocumentLM import DocumentLM
from ParsimoniousLM import ParsimoniousLM
import logging
import os
from nltk.stem import WordNetLemmatizer
import gensim
import nltk
import sys
import string




# we want to log the process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class TR(object):
    def __init__(self, ldaPath, numTopics, mu, threshold, numIteration):
        self.ldaPath = ldaPath
        #self.outPutPath = outPutPath
        self.numTopics = numTopics
        self.mu = mu
        self.threshold = threshold
        self.numIteration = numIteration
    def runTR(self):
        lda = gensim.models.LdaModel.load(os.path.join(self.ldaPath,"lda.model"))
        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.ldaPath,"mtsamples.dict"))
        topics = {}
        for i in range(0, self.numTopics):
            topics[i] = lda.show_topic(i, topn=200)
        #CM = CollectionLM("", topics)
        CM = CollectionLM("", "", topics, "")
        CM.buildTMcollectionLM();
        terms = {}
        for i in range(0, len(lda.state.sstats[1])):
            terms[lda.id2word[i]] = i

        k = 0
        for i in range(0, self.numTopics):
            topic = lda.show_topic(i, topn=200)
            TLM = DocumentLM("", "", topic, "", CM.vocab)
            TLM.buildTopicLM()
            PLM = ParsimoniousLM(CM.vocab, TLM.LM, TLM.termFreq, CM.LM, self.mu, self.threshold, self.numIteration)
            #PLM = ParsimoniousLM(TLM.LM, CM.vocab, TLM.termFreq, CM.LM, self.mu, self.threshold, self.numIteration)
            PLM.parsimonize()
            lda.state.sstats[k] = 0
            lda.expElogbeta[k] = 0
            lda.state.eta = 0
            #ind = 0
            for tok in PLM.vocab.keys():
                if tok in terms:
                    lda.state.sstats[k][terms[tok]] = PLM.docLM[PLM.vocab[tok]]
                    lda.expElogbeta[k][terms[tok]] = PLM.docLM[PLM.vocab[tok]]
                #ind += 1
            k += 1
        lda.save(os.path.join(self.ldaPath, "lda-TR.model"))
            

if __name__ == "__main__":

    ldaPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    outPutPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    tr = TR(ldaPath, outPutPath, 10, 0.5, 0.001, 10)
    tr.runTR()    




