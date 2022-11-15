from nltk.stem import WordNetLemmatizer
import numpy as np
import re

scraped_l = np.array(finaldf_['Scraped Lyrics'])

removelines = ['Chorus x2','x2','repeat 2x',' repeat 2x','chorus repeat','chorus repeat and fade','Repeat chorus 3rd verse',
               'Repeat bridge chorus','Repeat chorus last verse','Repeat chorus 5th verse chorus','REPEAT CHORUS fading away',
               'PostChorus x3','PreChorus','PostChorus','Postchorus','CHORUS  as before', 'Repeat 3rd verse',
               '1st verse Oo  backing vocals on each line','2nd verse Ah  backing vocal on each line',
               'Guitar solo coming','Here comes the guitar solo ', '']
all_songs = []
lmz = WordNetLemmatizer()

for i in np.arange(0, len(scraped_l), 1): # songs
    new_song_lines = scraped_l[i].split('|')
    new_song_fin = []
    for j in np.arange(0, len(new_song_lines), 1): # lines
        if new_song_lines[j] not in removelines:
            new_line = []
            all_words = new_song_lines[j].split()
            for k in np.arange(0, len(all_words), 1): # words
                new_word = all_words[k]
                new_word_1 = re.sub('[^a-zA-Z\']', '', new_word)
                new_word_2 = new_word_1.lower()
                new_word_3 = lmz.lemmatize(new_word_2)
                new_line.append(new_word_3)
            joined_line = " ".join(new_line) # Needed for Word2Vec sentences
            new_song_fin.append(joined_line)
    all_songs.append(new_song_fin)
    
finaldf_['Word2Vec Sentences'] = all_songs
