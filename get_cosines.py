# -*- coding: utf-8 -*-
import os
import sys
import string
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import pandas as pd
from scipy.spatial.distance import cosine
from nltk.stem.porter import PorterStemmer
import math

stemmer = PorterStemmer()
FREQ_THRESH = 200000
NO_SENTS = 4784808
vocab = pickle.load( open("vocab_dict.pkl", "rb" ) )

common_english_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
                 "for", "not", "on","with", "he", "as", "you", "do","at","this","but",
                 "his", "by","from","they","we","say","her","she","or","an","will","my",
                 "one", "all","would","there","their","what","so","if","about","who",
                 "which", "me", "when", "go","make","can","like","no","him","your","could",
                 "them", "other","than","then","only","its","also","after","how","our",
                 "well","way","want","because","any","these","most","into","up","problem"]

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
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def countTerm(term):
    if vocab.has_key(term):
        vocab[term] += 1
    else:
        vocab[term] = 1

def computeTFsaccross(docs):

    index = 0
    for q in docs:

        tokens = tokenize(q)

        for token in tokens:
            countTerm(token)

        index += 1
        if index % 100000 == 0:
            print(">>>>>>>> Generated data for {} sents... ".format(index))
            print(">>>>>>>> Vocab length is {} terms... ".format(len(vocab)))
            print(">>>>>>>> Memory of the vocab this far {} ... ".format(float(sys.getsizeof(vocab)) / 1024 / 1024))

    with open("vocab_dict.pkl", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def getFrqMatrix(curr_pair):
    try:
        # 3.2. get their TF vectors
        curr_pair_tf_mtx = CountVectorizer(tokenizer=tokenize, stop_words=common_english_words).fit_transform(
            curr_pair).todense()
        # 3.3 get their joint feature names
        terms_union = CountVectorizer(tokenizer=tokenize, stop_words=common_english_words).fit(
            curr_pair).get_feature_names()

    except:
        try:
            # 3.2. get their TF vectors
            curr_pair_tf_mtx = CountVectorizer(tokenizer=tokenize).fit_transform(curr_pair).todense()
            # 3.3 get their joint feature names
            terms_union = CountVectorizer(tokenizer=tokenize).fit(curr_pair).get_feature_names()

        except:

            curr_pair_tf_mtx = []
            terms_union = []
            pass
    return curr_pair_tf_mtx, terms_union


if __name__ == '__main__':

    #show("STARTING")

    train1 = pd.read_csv("data/train.csv")
    #test1 = pd.read_csv("data/test.csv")

    index = 0
    tf_idf_cosine = []

    for i in range(train1.shape[0]):

        # 3. for each pair
        # 3.1. make them a list
        q1 = map(lambda x: str(x).lower().translate(None, string.punctuation), [train1.loc[i, "question1"]])
        q2 = map(lambda x: str(x).lower().translate(None, string.punctuation), [train1.loc[i, "question2"]])

        curr_pair = q1
        curr_pair.extend(q2)

        curr_pair_tf_mtx, terms_union = getFrqMatrix(curr_pair)

        # 3.4 get the IDF vecs from the - all sents idfs list
        idfs = [vocab.get(key) for key in terms_union]

        for i in range(len(idfs)):
            if idfs[i] == None :
                idfs[i] = 0

        # print terms_union

        ## 3.4.1 filter which terms have freq lower than thr
        filtered_terms_indx = [idfs.index(x) for x in idfs if x < FREQ_THRESH]

        curr_pair_tf_mtx = curr_pair_tf_mtx[:, filtered_terms_indx]
        terms_union = [terms_union[x] for x in filtered_terms_indx]
        idfs = [idfs[x] for x in filtered_terms_indx]
        # print terms_union


        if len(idfs) > 0:
            ## transform IDF
            try:
                idfs = [math.log(float(NO_SENTS) / (key +1)) for key in idfs]
                # 3.5 multiply idfs with tf
                curr_pair_tf_mtx = np.array(curr_pair_tf_mtx) * np.array(idfs).tolist()
                tf_idf_cosine.append(cosine(curr_pair_tf_mtx[0], curr_pair_tf_mtx[1]))

            except:
                #print(idfs)
                tf_idf_cosine.append([1])

                pass

        else:
            tf_idf_cosine.append([1])

        index += 1
        if index % 10000 == 0:
            print(">>>>>>>> Generated data for {} sents... ".format(index))
            print(">>>>>>>> Generated data for {} sents... ".format(len(tf_idf_cosine)))

    print(">>>>>>>> Generated data for {} sents... ".format(len(tf_idf_cosine)))
    with open("Train_cosines.pkl", "wb") as f:
        pickle.dump(tf_idf_cosine, f, pickle.HIGHEST_PROTOCOL)
    #np.save("Train_cosines.npy", np.array(tf_idf_cosine))

