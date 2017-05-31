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

class TM(object):

    def __init__(self, corpus, dictionary, MODELS_DIR, NUM_TOPICS):
        #self.text_dir = collectionPath
        self.num_topics = NUM_TOPICS
        self.model_dir = MODELS_DIR
        self.corpus = corpus
        self.dictionary = dictionary

    def _run_LDA(self):
        stoplist = set(nltk.corpus.stopwords.words("english"))
        corpus = MyCorpus(self.text_dir, stoplist)

        corpus.dictionary.save(os.path.join(self.model_dir, "mtsamples.dict"))
        gensim.corpora.MmCorpus.serialize(os.path.join(self.model_dir, "mtsamples.mm"), 
                                          corpus)
        # Run lda
        lda = gensim.models.LdaModel(corpus, id2word=corpus.dictionary, num_topics=self.num_topics, iterations=1000, passes=50)
        # save the model
        lda.save(os.path.join(self.model_dir, "lda.model"))

    def run_LDA(self):
        # Run lda
        lda = gensim.models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=self.num_topics, iterations=1000, passes=10)

        # save the model
        lda.save(os.path.join(self.model_dir, "lda.model"))


def iter_docs(topdir, stoplist):
    remove = dict.fromkeys(map(ord, '\n ' + string.punctuation))
    for f in os.listdir(topdir):
        currDir = os.path.join(topdir, f)
        if os.path.isdir(currDir):
            for fn in os.listdir(os.path.join(topdir,f)):
                if fn != ".DS_Store":
                    fin = open(os.path.join(currDir, fn), 'rb')
                    text = fin.read().decode('utf-8', 'ignore') 
                    fin.close()
                    x = (x.translate(remove) for x in nltk.word_tokenize(text.lower()))
                    yield x


def iter_docs_simple(topdir, stoplist):
    remove = dict.fromkeys(map(ord, '\n ' + string.punctuation))
    for f in os.listdir(topdir):
        currDir = os.path.join(topdir, f)
        if os.path.isdir(currDir):
            for fn in os.listdir(os.path.join(topdir,f)):
                if fn != ".DS_Store":
                    fin = open(os.path.join(currDir, fn), 'rb')
                    text = fin.read().decode('utf-8', 'ignore') 
                    fin.close()
                    yield (x for x in
                       gensim.utils.tokenize(text, lowercase=True, deacc=True,
                                             errors="ignore")
                       )


                    
# transfering the documents to bow format
class MyCorpus(object):

    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs_simple(topdir, stoplist))
        
    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)


if __name__ == "__main__":
    modelDir = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    dictionary = gensim.corpora.Dictionary.load(os.path.join(modelDir,"mtsamples.dict"))
    corpus = gensim.corpora.MmCorpus(os.path.join(modelDir,"mtsamples.mm"))
    tm = TM(corpus, dictionary, modelDir, 20)
    tm.run_LDA()
    #TEXTS_DIR = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened"
    #MODELS_DIR = "/Users/admin/Downloads/20_newsgroups/Preprocessed-lemmas-shortened-models"
    #tm = TM(TEXTS_DIR, MODELS_DIR, 10)
    #tm.run_LDA()





