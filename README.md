# HiTR

This is the source code for the following papers:

Hosein Azarbonyad, Mostafa Dehghani, Tom Kenter, Maarten Marx, Jaap Kamps, and Maarten de Rijke, "Hierarchical Re-estimation of Topic Models for Measuring Topical Diversity", In Proceedings of the 39th European Conference on Information Retrieval (ECIR’17), 2017 

Hosein Azarbonyad, Mostafa Dehghani, Tom Kenter, Maarten Marx, Jaap Kamps, and Maarten de Rijke, "HiTR: Hierarchical Topic Model Re-estimation for Measuring Topical Diversity of Documents", To appear in IEEE Transactions on Knowledge and Data Engineering (TKDE)

The goal is to remove generality and impurity from topic models. General topics only include common information from a background corpus and are assigned to most of the documents in the collection. Impure topics contain words that are not related to the topic. A hierarchical re-estimation approach for topic models is developed to combat generality and impurity; the method operates at three levels: words, topics, and documents:

1. Document Re-estimation (DR) re-estimates the language model per document P (w | d) and removes general words from documents before training topic models; 
2. Topic Re-estimation (TR) re-estimates the language model per topic P (w | t) and removes general words from trained topics; 
3. Topic Assignment Re-estimation (TAR) re-estimates the distribution over topics per document P (t | d) and removes general topics from documents.

# Usage

An example on how to run the code is given in HiTR.py. Depending on the combination of the mentioned re-estimation approaches that you want to use, you can disable or enable them. 

## Input
1. corpus: the corpus should be in document_id-term_id matrix format, known as .mm in Gensim. This is the same as the input format for gensim.models.LdaModel.
2. dictionary: the dictionary is the same as gensim.corpora.Dictionary (in .dict format).
3. ldapath: the path to where the lda model will be saved.
4. outPutPath: the path to where the topic-assignments will be saved.

## Output
1. The re-estimated document-term matrix and will be stored in mtsamples-tmp.mm.
2. The re-estimated topics will be saved as lda-TR.model.
3. The re-estimated topic-assignments will be returned after running TAR step. 
