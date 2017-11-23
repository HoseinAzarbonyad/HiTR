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
from operator import truediv
import pickle

logger = logging.getLogger(__name__)

# we want to log the process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class Stats(object):
    def __init__(self, corpus, dictionary, outPutPath):
        #logger.info("Running HiTR")
        #self.documentsPath = documentsPath
        self.corpus = corpus
        #self.ldaPath = ldaPath
        self.outPutPath = outPutPath
        #self.numTopics = numTopics
        #self.mu = mu
        #self.threshold = threshold
        #self.numIteration = numIteration
        self.dictionary = dictionary
        
    def calcStats(self):
        logger.info("Running DR")
        mu = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #mu = [0.8]
        threshold = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        stats = {}
        vocabFile = open(os.path.join(self.outPutPath, "docVocabSizes-all.txt"), 'w')
        typeTokenFile = open(os.path.join(self.outPutPath, "docTypeTokenRatios-all.txt"), 'w')
        p_movedFile = open(os.path.join(self.outPutPath, "p_moved-all.txt"), 'w')
        for m in mu:
            stats[m] = {}
            for th in threshold:
                if os.path.isfile("mtsamples-tmp.mm"):
                    os.remove("mtsamples-tmp.mm")
                if os.path.isfile("mtsamples-tmp.mm.index"):
                    os.remove("mtsamples-tmp.mm.index")
                modelPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
                dictionary = gensim.corpora.Dictionary.load(os.path.join(modelPath,"mtsamples.dict"))
                corpus = gensim.corpora.MmCorpus(os.path.join(modelPath,"mtsamples.mm"))
                print "DR for: " + str(m) + " " + str(th)
                dr = DR(corpus, dictionary, m, th, 20)
                dr.runDR()  
                vocabSize = self.calcVocabSize(dr.corpus)
                avgDocVocabSize, docVocabSizes = self.calcDocVocabSize(dr.corpus)
                avgTypeTokenRatio, docTypeTokenRatios  = self.calcTypeTokenRatio(dr.corpus)
                avgp_moved, p_moved = self.calcp_moved(corpus, dr.corpus)
                print str(avgDocVocabSize) + " " + str(avgTypeTokenRatio) + " " + str(avgp_moved)
                del dr
                line = ""
                line += str(m) + " " + str(th) + " "
                for l in docVocabSizes:
                    line += str(l) + " "
                vocabFile.write(line + "\n")

                line = ""
                line += str(m) + " " + str(th) + " "
                for l in docTypeTokenRatios:
                    line += str(l) + " "
                typeTokenFile.write(line + "\n")

                line = ""
                line += str(m) + " " + str(th) + " "
                for l in p_moved:
                    line += str(l) + " "
                p_movedFile.write(line + "\n")

                stats[m][th] = []
                stats[m][th].append(vocabSize)
                stats[m][th].append(avgDocVocabSize)
                stats[m][th].append(avgTypeTokenRatio)
                stats[m][th].append(avgp_moved)
                del corpus
                del dictionary

        with open('res-all.txt', 'wb') as handle:
            pickle.dump(stats, handle)
        vocabFile.close()
        typeTokenFile.close()
        p_movedFile.close()

    def calcVocabSize(self, corpus):
        numTerms = 0
        uniqueTerms = {}
        for doc in corpus:
            for token in doc:
                if token[0] not in uniqueTerms:
                    uniqueTerms[token[0]] = 1
                    numTerms += 1
        return numTerms

    def calcDocVocabSize(self, corpus):
        avgDocVocabSize = 0
        docVocabSizes = [len(doc) for doc in corpus]
        avgDocVocabSize = float(sum([len(doc) for doc in corpus])) / len(corpus)
        return avgDocVocabSize, docVocabSizes

    def calcTypeTokenRatio(self, corpus):
        uniqueTerms = {}
        corpusSize = 0
        numTerms = 0
        for doc in corpus:
            for token in doc:
                if token[0] not in uniqueTerms:
                    uniqueTerms[token[0]] = 1
                    numTerms += 1
            corpusSize += sum([t[1] for t in doc])
        avgTypeTokenRatio = 0
        if corpusSize > 0:
            avgTypeTokenRatio = float(len(uniqueTerms)) / corpusSize

        #docTypeTokenRatios = map(truediv, [len(doc) for doc in corpus], [sum([t[1] for t in doc]) for doc in corpus])
        a = [len(doc) for doc in corpus]
        b = [sum([t[1] for t in doc]) for doc in corpus]
        docTypeTokenRatios = [x/y if y else 0 for x,y in zip(a,b)]
        return avgTypeTokenRatio, docTypeTokenRatios

    def calcp_moved(self, corpus1, corpus2):
        docId = 0
        p_moved = []
        for doc in corpus1: 
            words = {}
            for token in corpus2[docId]:
                words[token[0]] = token[1]
            sumFreq = 0
            docLen = 0
            for token in doc:
                docLen += token[1]
                if token[0] not in words:
                    sumFreq += token[1]
            p = 0
            if docLen > 0:
                p = float(sumFreq) / docLen
            #p_moved.append(float(sumFreq) / docLen)
            p_moved.append(p)
            docId += 1
        avgp_moved = sum(p_moved) / len(p_moved)
        return avgp_moved, p_moved
        
if __name__ == "__main__":
    modelPath = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(modelPath,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(modelPath,"mtsamples.mm"))

    outPutPath = "/Users/admin/Downloads/20_newsgroups/stats"

    stat = Stats(corpus, dictionary, outPutPath)
    stat.calcStats()
