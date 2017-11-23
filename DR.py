from CollectionLM import CollectionLM
from DocumentLM import DocumentLM
from ParsimoniousLM import ParsimoniousLM
import logging
import os
import gensim
import numpy as np



logger = logging.getLogger(__name__)

class DR(object):

	def __init__(self, corpus, dictionary, mu, threshold, numIteration):
		#self.collectionPath = collectionPath
		self.mu = mu
		self.threshold = threshold
		self.numIteration = numIteration
		self.corpus = corpus
		self.dictionary = dictionary


	def _runDR(self):

		colPath = os.path.join(self.collectionPath, "docs")
		CLM = CollectionLM(colPath)
		regenPath = os.path.join(self.collectionPath, "tmp")
		if not os.path.exists(regenPath):
			os.makedirs(regenPath)

		for d in os.listdir(colPath):
			path = os.path.join(colPath, d)
			DLM = DocumentLM(path)
			DLM.buildDocumentLMFromFile()
			docLen = len(DLM.LM)
			PLM = ParsimoniousLM(DLM.LM, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
			PLM.parsimonize()
			self.regenerateDoc(d, DLM.LM, docLen, regenPath)
			#print PLM.docLM

	def docDR(self, docId, DLM):
		docLen = np.sum(DLM.LM)
		print "lm before: "
		print DLM.LM[DLM.LM > 0]
		print np.count_nonzero(DLM.LM)
		PLM = ParsimoniousLM(DLM.LM, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
		PLM.parsimonize()
		print "PLM lm len: ", np.count_nonzero(PLM.docLM)
		print "lm after: "
		print PLM.docLM[PLM.docLM > 0]
		self.regenerateDoc(docId, PLM.docLM, docLen)
		return PLM

	def runDR(self):

		CLM = CollectionLM(self.corpus, self.dictionary, "", "")
		CLM.buildCorpusLM()
		docId = 0
		new_corpus = []
		for doc in self.corpus:
			DLM = DocumentLM(doc, self.dictionary, "", "", "")
			DLM.buildDocumentLM()
			docLen = sum([w[1] for w in doc])
			#print DLM.LM[DLM.LM > 0]
			PLM = ParsimoniousLM(CLM.vocab, DLM.LM, DLM.termFreq, CLM.LM, self.mu, self.threshold, self.numIteration)
			PLM.parsimonize()
			#print PLM.docLM[PLM.docLM > 0]
			new_doc = self.regenerateDoc(docId, PLM.docLM, docLen)
			docId += 1
			#if len(doc) > 0:
			new_corpus.append(new_doc)
			del DLM
			del PLM
			#print PLM.docLM
		del self.corpus
		gensim.corpora.MmCorpus.serialize("mtsamples-tmp.mm", new_corpus)
		self.corpus = gensim.corpora.MmCorpus("mtsamples-tmp.mm")
		print self.corpus[10]
		print len(self.corpus)
		del CLM

	def regenerateDoc(self, docId, LM, docLen):
		index = 0
		doc = []
		for w in self.corpus[docId]:
			if LM[w[0]] > 0:
				freq = int(LM[w[0]] * docLen)
				tup = (w[0], freq)
				#doc[index] = tup
				doc.append(tup)
				index += 1
		return doc
		

if __name__ == "__main__":
    modelDir = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(modelDir,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(modelDir,"mtsamples.mm"))
    CLM = CollectionLM(corpus, dictionary, "", "")
    CLM.buildCorpusLM()

    DLM = DocumentLM(corpus[10], dictionary, "", "", "")
    DLM.buildDocumentLM()

    #rows = np.nonzero(DLM.LM)
    #print DLM.LM[rows]
    print np.count_nonzero(DLM.LM)
    print corpus[10]
    dr = DR(corpus, dictionary, 0.5, 0.01, 5)
    dr.runDR()
    

        


