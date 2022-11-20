from nltk import word_tokenize
from gensim.models import Word2Vec
import numpy as np

##Concatenate songs, keeping line separated
sentences_fin = []
for s in np.array(finaldf_['Word2Vec Sentences']):
    for l in s:
        sentences_fin.append(l)

##Tokenize word in each line
sentences_fin = [word_tokenize(line) for line in sentences_fin]

##Model
model = Word2Vec(sentences = sentences_fin, min_count=3, sg=1, hs=1, window=7, vector_size=300)
