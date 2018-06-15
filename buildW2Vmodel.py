# -*- coding: utf-8 -*-
import gc
from gensim.models import word2vec
from timeit import default_timer as timer
from settings import *
import string
import numpy as np
import pickle
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

max_sent_length = map(lambda x:-x, range(500))

common_english_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
                 "for", "not", "on","with", "he", "as", "you", "do","at","this","but",
                 "his", "by","from","they","we","say","her","she","or","an","will","my",
                 "one", "all","would","there","their","what","so","if","about","who",
                 "which", "me", "when", "go","make","can","like","no","him","your","could",
                 "them", "other","than","then","only","its","also","after","how","our",
                 "well","way","want","because","any","these","most","into","up"]


def stem_tokens(tokens, stemmer):
    stemmed = []

    for item in tokens:
        stemmed_item = item
        try:
            stemmed_item = stemmer.stem(item)
        except Exception:
            print("EXCEPTION WHILE STeM")
            print(stemmed_item)
            pass
        stemmed.append(stemmed_item)

    return stemmed


def tokenize(text):

    tokens = [x for x in nltk.word_tokenize(text) if x not in common_english_words]
    if len(tokens) == 0:
        tokens = [x for x in nltk.word_tokenize(text)]

    for i in range(len(tokens)):
        if len(''.join(e for e in tokens[i] if e.isalnum())) > 0:
            tokens[i] = ''.join(e for e in tokens[i] if e.isalnum())

    stems = stem_tokens(tokens, stemmer)

    if len(tokens) > min(max_sent_length):
        max_sent_length[max_sent_length.index(min(max_sent_length))] = len(tokens)
    return stems


def show(message):
    print(" ")
    print(np.repeat("*", 50).tostring())
    print("   " + message)
    print(np.repeat("*", 50).tostring())


def generateWord2VecTrainData():

    test1 = pd.read_csv(TEST_FILE)
    train1 = pd.read_csv(TRAIN_FILE)


    print(">>>>>>>> Train data size is  {} ... ", train1.shape)
    print(">>>>>>>> Test data size is  {} ... ", test1.shape)

    training_sents = []

    qus = map(lambda x: str(x).lower().translate(None, string.punctuation), train1.question1.values.tolist())
    qus.extend(map(lambda x: str(x).lower().translate(None, string.punctuation), train1.question2.values.tolist()))
    qus.extend(map(lambda x: str(x).lower().translate(None, string.punctuation), test1.question1.values.tolist()))
    qus.extend(map(lambda x: str(x).lower().translate(None, string.punctuation), test1.question2.values.tolist()))


    all_sents = list(set(qus))

    print(">>>>>>>> Working corpora size of {} ... ".format(len(all_sents)))

    index = 0
    for sent in all_sents:

        tokens = tokenize(sent)

        if len(tokens) > 0:
            training_sents.append(tokens)

        index += 1
        if index % (100*MAGNITUDE) == 0:
            print(">>>>>>>> Generated data for {} sents... ".format(str(index)))
            print(sorted(max_sent_length))

    print("Generated {} samples and now saving to files".format(len(training_sents)))

    with open(WORD2VEC_TRAIN_DATA , 'wb') as data_file:
        pickle.dump(training_sents, data_file)

def trainW2Vmodel():

    # sg defines the training algorithm.By default(sg=0), CBOW is used.Otherwise(sg=1), skip - gram is employed.
    # min_count = how many times a word needs to appear in corpus to be included in the vocab, default  = 5
    # size =  # of NN hidden nodes - default value is 100
    # max_vocab_size = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM
    # sample = threshold for configuring which higher-frequency words are randomly downsampled; default is 1e-3, useful range is (0, 1e-5).
    # full list of params is here: http://radimrehurek.com/gensim/models/word2vec.html
    # gensim.models.word2vec.Word2Vec
    # window is the maximum distance between the current and predicted word within a sentence.
    # workers


    with open(WORD2VEC_TRAIN_DATA, 'rb') as f:
        training_sents = pickle.load(f)

    print("Read {} samples and now building model".format(len(training_sents)))

    encoding_model = word2vec.Word2Vec(training_sents, size=WORD_ENC_SIZE, window=MAX_WORD_WINDOW_DIST, min_count=1, workers = NCPU, sg=1, sample=0.01)

    print(">>>>>>>>  ... saving model to disk ...")
    encoding_model.save(WORD2VEC_MODEL_LOC)


if __name__ == '__main__':


    start = timer()
    show(" Generating train data for w2v model... ")
    generateWord2VecTrainData()
    print("Elapsed time: {}".format(timer()- start))

    start = timer()
    show("Generating w2v model ... ")
    trainW2Vmodel()
    print("Elapsed time: {}".format(timer()- start))




