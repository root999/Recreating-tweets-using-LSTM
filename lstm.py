"""
Twitter'dan gerekli API izinlerini alamadığım için, twitter verileri 
kişisel arşivin istenmesiyle elde edilmiştir. Javascript kodu içerisinde
liste içerisindeki dictionary'lerde elde edilen twitter verileri bir
python dosyasına atılmış,
handleTwitter class'ı yardımıyla emoji gibi karakterlerden arındırılmış,
aynı zamanda sadece tweet içeriğinin bulunduğu txt dosyaya dönüştürülmüştür.

Verinin ilk hali aşağıdaki örnekteki gibidir.

data = [ {
  "tweet" : {
    "retweeted" : False,
    "source" : "<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>",
    "entities" : {
      "hashtags" : [ ],
      "symbols" : [ ],
      "user_mentions" : [ ],
      "urls" : [ ]
    },
    "display_text_range" : [ "0", "52" ],
    "favorite_count" : "6",
    "id_str" : "1268833353513517063",
    "truncated" : False,
    "retweet_count" : "0",
    "id" : "1268833353513517063",
    "created_at" : "Fri Jun 05 09:13:39 +0000 2020",
    "favorited" : False,
    "full_text" : "Ksksksksksksk aniden karşına çıkan yasak iptali şoku",
    "lang" : "tr"
  }
}]

"""

from selindata import data
import unicodedata
from unidecode import unidecode
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
def sample(preds,diversity):
    preds = np.asarray(preds).astype('float64')  
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class handleTwitter():
    def __init__(self,filename):
        self.data = self.createData()
        self.no_emoji_data = self.noEmoji()
        self.writedata(filename)
    def createData(self):
        tweets = []
        for dt in data:
            if('http' not in dt['tweet']['full_text'] and '&lt'not in dt['tweet']['full_text'] \
                                            and '@' not in dt['tweet']['full_text']):
                tweets.append(dt['tweet']['full_text'])
        return tweets

    def deEmojify(self,inputString):
        returnString = ""
        for character in inputString:
            try:
                character.encode("ascii")
                returnString += character
            except UnicodeEncodeError:
                replaced = unidecode(str(character))
                if replaced != '':
                    returnString += replaced
                else:
                    try:
                        returnString += "[" + unicodedata.name(character) + "]"
                    except ValueError:
                        returnString += "[x]"

        return returnString
    def noEmoji(self):
        emojifree = []
        for t in self.data:
            no_emj= self.deEmojify(t)
            if '[' not in no_emj:
                no_emj = no_emj.lower()
                emojifree.append(no_emj)
        return emojifree
    def writedata(self,filename):
        with open(filename, "w") as f:
            for s in self.no_emoji_data:
                f.write(str(s) +"\n")

class preProcessor():

    def __init__(self,filename):
        self.NUM_OF_SEQ = None
        self.MAX_LEN = 40
        self.SEQ_JUMP = 3
        self.CORPUS_LENGHT = None
        self.corpus = self.createCorpus(filename)
        self.chars = sorted(list(set(self.corpus)))
        self.NUM_OF_CHARS = len(self.chars)
        self.char_to_idx,self.idx_to_char = self.createIndices()
        self.sequences,self.next_chars = self.createSequences()
        self.dataX,self.dataY = self.one_hot()
       
    def getTweets(self,filename):
        tweets = []
        with open(filename, "r") as f:
            for line in f:
                tweets.append(line.strip())
        return tweets

    def createCorpus(self,filename):
        tweets = self.getTweets(filename)
        corpus = u' '.join(tweets)
        self.CORPUS_LENGHT= len(corpus)
        return corpus

    def createIndices(self):
        char_to_idx = {}
        idx_to_char = {}
        for i,c in enumerate(self.chars):
            char_to_idx[c]=i
            idx_to_char[i]=c
        return char_to_idx,idx_to_char

    def createSequences(self):
        sequences = []
        next_chars = []
        for i in range(0,self.CORPUS_LENGHT-self.MAX_LEN,self.SEQ_JUMP):
            sequences.append(self.corpus[i: i+self.MAX_LEN])
            next_chars.append(self.corpus[i+self.MAX_LEN])
        self.NUM_OF_SEQ = len(sequences)
        return sequences,next_chars

    def one_hot(self):
        dataX = np.zeros((self.NUM_OF_SEQ,self.MAX_LEN,self.NUM_OF_CHARS),dtype=np.bool)
        dataY = np.zeros((self.NUM_OF_SEQ,self.NUM_OF_CHARS),dtype=np.bool)
        for i,seq in enumerate(self.sequences):
            for j,c in enumerate(seq):
                dataX[i,j,self.char_to_idx[c]]=1
            dataY[i,self.char_to_idx[self.next_chars[i]]]=1
        return dataX,dataY

class LSTModel():
    def __init__(self,max_len,num_of_chars,preprocessor):
        self.max_len = max_len
        self.num_of_chars = num_of_chars
        self.model = self.createModel()
        self.preprocessor = preprocessor

    def createModel(self,layer_size = 128,dropout=0.2,learning_rate=0.01,verbose=1):
        model = Sequential()
        model.add(LSTM(layer_size,return_sequences = True,input_shape=(self.max_len,self.num_of_chars)))
        model.add(Dropout(dropout))
        model.add(LSTM(layer_size, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(self.num_of_chars, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
        if verbose:
            print('Model Summary:')
            model.summary()
        return model

    def trainModel(self,X, y, batch_size=128, nb_epoch=60, verbose=0):
        checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='loss', verbose=verbose, save_best_only=True, mode='min')
        history = self.model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[checkpointer])
        return history
    
    def createTweets(self,num_of_tweets=10,tweet_length=70):
        f=open("produced_tweets.txt", "a+")
        self.model.load_weights('weights.hdf5')
        tweets = []
        seq_starts =[]
        diversities = [0.2, 0.5,0.1]
        for i,char in enumerate(self.preprocessor.corpus):
            if char == ' ':
                seq_starts.append(i)
        for div in diversities:
            f.write("---- diversity : %f\n"% div)
            for i in range(num_of_tweets):
                f.write("---- Tweet %d:\n" % i)
                begin = np.random.choice(seq_starts)
                tweet = u''
                sequence = self.preprocessor.corpus[begin:begin+self.preprocessor.MAX_LEN]
                tweet += sequence
                f.write("---Random Sequence beginning: %s\n" % tweet)
                for _ in range(tweet_length):
                    input_data = np.zeros((1,self.preprocessor.MAX_LEN,self.preprocessor.NUM_OF_CHARS),dtype=np.bool)
                    for t,char in enumerate(sequence):
                        input_data[0,t,self.preprocessor.char_to_idx[char]]=True
                    predictions = self.model.predict(input_data)[0]
                    next_idx = sample(predictions,div)
                    next_char = self.preprocessor.idx_to_char[next_idx]
                    tweet += next_char
                    sequence = sequence[1:] + next_char
                f.write("Generated using LSTM: %s\n" % tweet)
                #print(tweet)
                tweets.append(tweet)
        f.close()
        return tweets


        
if __name__ == "__main__":
    cwd = os.getcwd()
    filename = "deneme.txt"
    path = os.path.join(cwd,filename)
    if not os.path.exists(path):
        handler = handleTwitter(filename)
    preprocessor = preProcessor(filename)
    dataX = preprocessor.dataX
    dataY = preprocessor.dataY
    max_len = preprocessor.MAX_LEN
    num_of_chars = preprocessor.NUM_OF_CHARS
    lstm = LSTModel(max_len,num_of_chars,preprocessor)
    #history = lstm.trainModel(dataX,dataY,verbose=1,nb_epoch=120)
    tweets= lstm.createTweets()
    # f = open("loss.txt","w")
    # for i,loss_data in enumerate(history.history['loss']):
    #     msg_annotated = "{0}\t{1}\n".format(i, loss_data)
    #     f.write(msg_annotated)
    # f.close()
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(preprocessor.sequences)
    Xval = vectorizer.transform(tweets)
    # print(str(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(axis=1).mean()))
    # f = open("pairwise_dist.txt","w")
    # f.write(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(axis=1).mean())
    # f.close()




    












