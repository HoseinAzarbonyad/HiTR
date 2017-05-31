from collections import defaultdict
from heapq import nlargest
import logging
import os
from collections import Counter
import nltk
import numpy as np

logger = logging.getLogger(__name__)

class ParsimoniousLM(object):
    
    
    def __init__(self, vocab, docLM, docTF, colLM, mu, threshold, numIteration):
        self.mu = mu
        self.threshold = threshold
        self.colLM = colLM
        self.numIteration = numIteration
        self.docTF = docTF
        self.docLM = docLM
        self.vocab = vocab
    def E_step(self):
        doc_terms = self.docLM > 0
        docLMW = self.mu * self.docLM[doc_terms]
        #self.docLM[doc_terms] = self.docTF[doc_terms] * docLMW / (docLMW + (1 - self.mu)*self.colLM[doc_terms])
        self.docLM[doc_terms] = self.div0(self.docTF[doc_terms] * docLMW, (docLMW + (1 - self.mu)*self.colLM[doc_terms]))
    
    def M_step(self):
        #self.docLM = self.docLM / sum(self.docLM)
        self.docLM = self.div0(self.docLM, sum(self.docLM))
        low_values_indices = self.docLM < self.threshold
        self.docLM[low_values_indices] = 0
        #self.docLM = self.docLM / sum(self.docLM)
        self.docLM = self.div0(self.docLM, sum(self.docLM))
        
    def normalizeLM(self):
        old_error_settings = np.seterr(divide='ignore')
        self.docLM = self.docLM / sum(self.docLM)
        
    def parsimonize(self):
        for i in range(self.numIteration):
            self.E_step()
            self.M_step()

    def div0(self, a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0  # -inf inf NaN
        return c