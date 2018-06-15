


# initial files locations
TEST_FILE = "data/test1.csv"
TRAIN_FILE = "data/train1.csv"
SAMPLE_SUB = "data/sub1.csv"

# locations for building word to vec model
WORD2VEC_TRAIN_DATA = "data/word2vec_train_data"
WORD2VEC_MODEL_LOC = "models/word_enc_model1.h5"

# locations of encoded data
TRAIN_Q1_SENTS = "data/train_q1_sents.npy"
TRAIN_Q2_SENTS = "data/train_q2_sents.npy"


TEST_Q1_SENTS = "data/test_q1_sents.npy"
TEST_Q2_SENTS = "data/test_q2_sents.npy"


# locations of final model validation and predictions
FINAL_MODEL = "models/final_model1.h5"
PREDICTIONS_FILE = "data/submission1.csv"
TEST_IDS = "data/test_ids.npy"
MODEL_RESULTS = "result.txt"

WORD_ENC_SIZE = 80
MAX_WORD_WINDOW_DIST = 5  # np.arange(3, 4,1)
word_encoding_vector_size = 81
max_sent_length = 30
batch_size = 200

# this is just for printing progress -> should be 500 when full data size used - not for grid
MAGNITUDE = 1

# cross-val folds and ratio - should be .1 or .2  -> not for grid
TEST_PERC = .2
FOLDS = 3 # should be say 10 - not for grid
PREDICTION_MODE = True
NO_SENTS = 4784808

import psutil
NCPU = psutil.cpu_count()

from params import *