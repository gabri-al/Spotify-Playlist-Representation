from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### Test Set Upload
_import_path = r'Your path'
testdf_ = pd.read_excel(_import_path, sheet_name = 'Spotify_Test')

# Encode target classes (1 = metal, 2 = pop)
genre_dic = {'Metal': 1, 'Pop': 0}

lyrics_clean1 = []; target_labels = []
# If a full line is like any in this list, remove it:
removelines = ['Chorus x2','x2','repeat 2x',' repeat 2x','chorus repeat','chorus repeat and fade','Repeat chorus 3rd verse',
               'Repeat bridge chorus','Repeat chorus last verse','Repeat chorus 5th verse chorus','REPEAT CHORUS fading away',
               'PostChorus x3','PreChorus','PostChorus','Postchorus','CHORUS  as before', 'Repeat 3rd verse',
               '1st verse Oo  backing vocals on each line','2nd verse Ah  backing vocal on each line',
               'Guitar solo coming','Here comes the guitar solo ', ''] 
for index, row in testdf_.iterrows():
    lines = row['Scraped Lyrics'].split('|')
    lines_final = [l for l in lines if l not in removelines]
    lines_final = " ".join(lines_final)
    lines_final = lines_final.strip().lower()
    lines_final = re.sub('[^a-zA-Z\']', ' ', lines_final)
    lyrics_clean1.append(lines_final)
    target_l = genre_dic[row['Genre']] # Apply encoding
    target_labels.append(target_l)

testdf_['Target Label'] = target_labels
testdf_['Lyrics Clean'] = lyrics_clean1

# Stem words and add in a df column
lmz = WordNetLemmatizer()
lemmatized_lyrics = []
for index, row in testdf_.iterrows():
    lyrics_clean = row['Lyrics Clean']
    lyrics_clean = lyrics_clean.split()
    ps = PorterStemmer()
    #lyrics_fin = [lmz.lemmatize(word) for word in lyrics_clean if word not in stop_words]
    lyrics_fin = [lmz.lemmatize(word) for word in lyrics_clean]
    lyrics_fin = ' '.join(lyrics_fin)
    lemmatized_lyrics.append(lyrics_fin)
    
testdf_['Lyrics Lemmatized'] = lemmatized_lyrics
del testdf_['Lyrics Clean']
print(testdf_.head(6))

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### Prepare train and test sets

# Compute avg vector for each song
def vectorize_songs(songs_list, word2vec_model):
    word_vectors = word2vec_model.wv # Extract word vectors from the word2vec model
    vectorized_list = []
    for i in np.arange(0, len(songs_list), 1):
        song_words = songs_list[i].split()
        new_song_vec = []
        for w in song_words:
            if w in word2vec_model.wv.key_to_index:
                new_song_vec.append(word_vectors[w])
        new_song_vec = np.array(new_song_vec)
        new_song_avg = np.average(new_song_vec, axis = 0)
        vectorized_list.append(new_song_avg)
    return vectorized_list

# Process both playlists
y_train = np.array(finaldf_['Target Label'])
y_test = np.array(testdf_['Target Label'])
X_train_lyrics = np.array(finaldf_['Lyrics Lemmatized'])
X_train = vectorize_songs(X_train_lyrics, model)
X_test_lyrics = np.array(testdf_['Lyrics Lemmatized'])
X_test = vectorize_songs(X_test_lyrics, model)

print(len(X_train))
print(len(X_test))

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### Logistic Regression Grid Search
parameters = {
    'penalty' : ['l1','l2','elasticnet'], 
    'C'       : np.logspace(-3,3,7),
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

LR = LogisticRegression(max_iter=1000)
LRCV = GridSearchCV(LR,                       # model
                   param_grid = parameters,   # hyperparameters
                   scoring='accuracy',        # metric for scoring
                   cv=10)                     # number of folds

LRCV.fit(X_train,y_train)
print("Tuned Hyperparameters :", LRCV.best_params_)
print("Accuracy :",LRCV.best_score_)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### Logistic Regression Train & Test
LogReg = LogisticRegression(max_iter=5000, penalty = 'l2', C = 1000.0, solver="newton-cg")
LogReg.fit(X_train,y_train)
y_pred = LogReg.predict(X_test)
print(y_pred) # Prediction

print("Train Accuracy:",LogReg.score(X_train, y_train))
print("Test Accuracy:",LogReg.score(X_test, y_test))
