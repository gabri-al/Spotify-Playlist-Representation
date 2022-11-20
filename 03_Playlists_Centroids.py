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

## Save centroid words
centroid_words_1 = []
for i in np.arange(0, len(word_vectors.similar_by_vector(centroid_1)), 1):
    word_ = word_vectors.similar_by_vector(centroid_1)[i][0]
    centroid_words_1.append(word_)

centroid_words_0 = []
for i in np.arange(0, len(word_vectors.similar_by_vector(centroid_0)), 1):
    word_ = word_vectors.similar_by_vector(centroid_0)[i][0]
    centroid_words_0.append(word_)

## Plot PCA on most representative words
import matplotlib.pyplot as plt
plt.style.use('bmh')
from sklearn.decomposition import PCA

reps_ = centroid_words_1.copy()
for w in centroid_words_0:
    reps_.append(w)
labels = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
reps_vec = []

word_vectors = model.wv
for w in reps_:
    reps_vec.append(word_vectors[w])

# Calculate PCA
pca = PCA(n_components = 10)
p1 = pca.fit_transform(reps_vec)
pl_1 = p1[:10]
pl_0 = p1[10:]
plt.scatter(pl_1[:,0],pl_1[:,1], c='darkblue', label='1-Metal')
plt.scatter(pl_0[:,0],pl_0[:,1], c='crimson', label='0-Pop')
plt.title("Most representative words : PCA")
plt.legend(prop={'size': 6})
plt.rcParams['figure.dpi'] = 100
plt.show()


## Plot PCA on centroids only
pca = PCA()
p1 = pca.fit_transform(np.array([centroid_1, centroid_0]))
pl_1 = p1[1]
pl_0 = p1[0]
plt.scatter(pl_1[0],pl_1[1], c='darkblue', label='1-Metal')
plt.scatter(pl_0[0],pl_0[1], c='crimson', label='0-Pop')
plt.title("Playlists Centroids | PCA")
plt.legend(prop={'size': 6})
plt.ticklabel_format(useOffset=False)
plt.show()
