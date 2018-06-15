# -*- coding: utf-8 -*-
import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import gc
import pandas as pd
from gensim.models import word2vec
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
from nltk.stem.porter import PorterStemmer
from settings import *
import string
import pickle

encoding_model = word2vec.Word2Vec.load(WORD2VEC_MODEL_LOC)

stemmer = PorterStemmer()

vocab = pickle.load( open( "vocab_dict.pkl", "rb" ) )

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

    return stems

def show(message):
    print(" ")
    print(np.repeat("*", 50).tostring())
    print("   " + message)
    print(np.repeat("*", 50).tostring())

def getWordEncoding(word, encoding_model):

    # word received may have any mixed cases !!

    ############# semantics vec

    try:
        skipgram_vect = encoding_model.wv[word]

    except Exception:
        print("EXCEPTION WHILE ENCODING")
        print(word)
        skipgram_vect = np.repeat(0, WORD_ENC_SIZE)
        pass

    w_tf = 1/NO_SENTS

    try:
        w_tf = vocab.get(word)/NO_SENTS
    except:
        print("EXCEPTION WHILE VOCAB LOOKUP")
        print(word)
        pass

    return np.append(skipgram_vect, w_tf)

def encodeSentence(sent, encoding_model):

    sent_encodings = []

    tokens = tokenize(sent)

    for i in range(len(tokens)):

        word_enc = getWordEncoding(tokens[i], encoding_model)
        sent_encodings.append(word_enc)

    # put one word in at least to ensure padding in case of empty list
    sent_encodings.append(np.repeat(0, word_encoding_vector_size))
    padded_X = pad_sequences([sent_encodings], maxlen=max_sent_length, dtype='float32', padding='post', truncating='pre', value=0)


    return padded_X

def process_question(q):

    word_features = encodeSentence(q, encoding_model)
    #print(word_features[0].shape)
    return word_features[0] #np.reshape(word_features, (word_features.shape[1], word_features.shape[2]))

def generateEncodedTrainingDataQ1():


    training_q1_sents = []

    train = pd.read_csv(TRAIN_FILE)
    print(">>>>>>>> Read train data size is  {} ... ", train.shape)

    questions = map(lambda x: str(x).lower().translate(None, string.punctuation), train.question1.values.tolist())

    train = None
    del train
    gc.collect()

    print(">>>>>>>> Generating training for pred model ... ")

    index = 0

    for q in questions:

        training_q1_sents.append(process_question(q))

        index += 1
        if index % (100*MAGNITUDE) == 0:
            print(">>>>>>>> Generated vector encodings training data for {} ... ".format(str(index)))

    training_q1_sents = np.array(training_q1_sents)


    print(">>>>>>>> The generated data shapes are as follows ... ")
    print(training_q1_sents.shape)


    np.save(TRAIN_Q1_SENTS, training_q1_sents)

def generateEncodedTrainingDataQ2():


    training_q2_sents = []

    train = pd.read_csv(TRAIN_FILE)
    print(">>>>>>>> Read train data size is  {} ... ", train.shape)

    questions = map(lambda x: str(x).lower().translate(None, string.punctuation), train.question2.values.tolist())

    train = None
    del train
    gc.collect()


    print(">>>>>>>> Generating training for pred model ... ")

    index = 0

    for q in questions:

        training_q2_sents.append(process_question(q))

        index += 1
        if index % (100*MAGNITUDE) == 0:
            print(">>>>>>>> Generated vector encodings training data for {} ... ".format(str(index)))


    training_q2_sents = np.array(training_q2_sents)

    print(">>>>>>>> The generated data shapes are as follows ... ")

    print(training_q2_sents.shape)

    np.save(TRAIN_Q2_SENTS, training_q2_sents)

def generateEncodedTestDataQ1():

    training_q1_sents = []

    train = pd.read_csv(TEST_FILE)

    print(">>>>>>>> Read train data size is  {} ... ", train.shape)

    questions = map(lambda x: str(x).lower().translate(None, string.punctuation), train.question1.values.tolist())

    train = None
    del train
    gc.collect()

    print(">>>>>>>> Generating training for pred model ... ")

    index = 0

    for q in questions:

        training_q1_sents.append(process_question(q))

        index += 1
        if index % (100*MAGNITUDE) == 0:
            print(">>>>>>>> Generated vector encodings training data for {} ... ".format(str(index)))

    training_q1_sents = np.array(training_q1_sents)

    print(">>>>>>>> The generated data shapes are as follows ... ")
    print(training_q1_sents.shape)

    np.save(TEST_Q1_SENTS, training_q1_sents)

def generateEncodedTestDataQ2():


    training_q2_sents = []

    train = pd.read_csv(TEST_FILE)

    print(">>>>>>>> Read train data size is  {} ... ", train.shape)

    questions = map(lambda x: str(x).lower().translate(None, string.punctuation), train.question2.values.tolist())

    train = None
    del train

    print(">>>>>>>> Generating training for pred model for {} items... ".format(len(questions)))

    index = 0

    for q in questions:

        training_q2_sents.append(process_question(q))

        index += 1
        if index % (100*MAGNITUDE) == 0:
            print(">>>>>>>> Generated vector encodings training data for {} ... ".format(str(index)))


    training_q2_sents = np.array(training_q2_sents)

    print(">>>>>>>> The generated data shapes are as follows ... ")

    print(training_q2_sents.shape)

    np.save("data/test_q2_part2.npy", training_q2_sents)

if __name__ == '__main__':

    #show("STARTING")

    start = timer()
    show("Generating train data ... ")
    generateEncodedTrainingDataQ1()
    generateEncodedTrainingDataQ2()
    print("Elapsed time: {}".format(timer()- start))
    start = timer()

    start = timer()
    show("Generating test data ... ")
    generateEncodedTestDataQ1()
    generateEncodedTestDataQ2()
    print("Elapsed time: {}".format(timer()- start))
    start = timer()
