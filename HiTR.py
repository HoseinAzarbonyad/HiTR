from CollectionLM import CollectionLM
from DocumentLM import DocumentLM
from ParsimoniousLM import ParsimoniousLM
from DR import DR
from TM import TM
from TR import TR
from TAR import TAR
import logging
import os
import gensim

logger = logging.getLogger(__name__)

# we want to log the process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class HiTR(object):
    def __init__(self, corpus, dictionary, ldaPath, outPutPath, numTopics):
        logger.info("Running HiTR")
        #self.documentsPath = documentsPath
        self.corpus = corpus
        self.ldaPath = ldaPath
        self.outPutPath = outPutPath
        self.numTopics = numTopics
        #self.mu = mu
        #self.threshold = threshold
        #self.numIteration = numIteration
        self.dictionary = dictionary
        
    def runHiTR(self):
        logger.info("Running DR")
        #dr = DR(os.path.join(self.documentsPath, "docs"), 0.5, 0.001, 10)
        dr = DR(corpus, dictionary, 0.5, 0.001, 10)
        dr.runDR()  

        logger.info("Running TM")
        #tm = TM(os.path.join(self.documentsPath, "tmp"), self.ldaPath, self.numTopics)
        tm = TM(corpus, dictionary, self.ldaPath, self.numTopics)
        tm.run_LDA()

        logger.info("Running TR")
        #tr = TR(self.ldaPath, self.outPutPath, self.numTopics, 0.5, 0.001, 10)
        tr = TR(self.ldaPath, self.numTopics, 0.5, 0.001, 10)
        tr.runTR()
        
        logger.info("Running TAR")
        #tar = TAR(self.ldaPath, os.path.join(self.documentsPath, "tmp"), self.outPutPath, 0.5, 0.001, 10)
        tar = TAR(self.corpus, self.ldaPath, "outputPath", 0.5, 0.001, 10)      
        topic_assignments = tar.runTAR()    
        logger.info("Done!")
        return topic_assignments
        
if __name__ == "__main__":
    documentsPath = "/Users/admin/Desktop/Projects/Topic Modeling"
    ldaPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    outPutPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(ldaPath,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(ldaPath,"mtsamples.mm"))

    outPutPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-TAR"

    hitr = HiTR(corpus, dictionary, ldaPath, outPutPath, 20)
    topics = hitr.runHiTR()
    print topics[0:10]

