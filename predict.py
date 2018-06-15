from settings import *
import numpy as np
from keras.models import load_model
import pandas as pd

def generatePredictions():

    model = load_model(FINAL_MODEL)

    Q12_sents = np.append(np.load(TRAIN_Q1_SENTS), np.load(TRAIN_Q2_SENTS), axis=1)
    sub = pd.read_csv(SAMPLE_SUB)

    print(">>>>>>>> Total data size ... ")
    print(Q12_sents.shape)

    sub.loc[:,"is_duplicate"] = model.predict(Q12_sents, batch_size=50000, verbose=1)

    next_start = 0

    for i in range(0):

        print(">>>>>>>> Working on part {}/3... ".format(i+1))

        Q1 = np.load("data/test_q1_part" + str(i+1) + ".npy")
        Q2 = np.load("data/test_q2_part" + str(i+1) + ".npy")

        Q12_sents = np.append(Q1,Q2, axis=1)

        Q1 = None
        Q2 = None

        l = Q12_sents.shape[0]

        print(">>>>>>>> Total data size ... ")
        print(Q12_sents.shape)

        print(">>>>>>>> To fill in sub from {} to {} / {}... ".format(next_start, (next_start + l-1), sub.shape[0]))

        sub.loc[next_start:(next_start + l-1),"is_duplicate"] = model.predict(Q12_sents, batch_size=50000 ,verbose=1)

        next_start +=  l

    ## save
    print(">>>>>>>> Saving predictions to disk ... ")
    sub.to_csv(PREDICTIONS_FILE, index = False)

if __name__ == '__main__':

    generatePredictions()