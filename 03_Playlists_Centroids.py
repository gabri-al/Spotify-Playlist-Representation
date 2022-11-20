# Extract word vectors from the word2vec model
word_vectors = model.wv

# Metal playlist
corpus_1_list = corpus_1.split() #from string to list

centroid_1 = np.average([word_vectors[w] for w in corpus_1_list if w in model.wv.key_to_index], axis=0)
print(word_vectors.similar_by_vector(centroid_1)) # Find the top-N most similar words to the centroid.

# Pop playlist
corpus_0_list = corpus_0.split()

centroid_0 = np.average([word_vectors[w] for w in corpus_0_list if w in model.wv.key_to_index], axis=0)
print(word_vectors.similar_by_vector(centroid_0))
