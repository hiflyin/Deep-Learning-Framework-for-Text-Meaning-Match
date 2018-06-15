
import os


MODEL_RESULTS = "result.txt"


if __name__ == '__main__':

    os.system('python buildW2Vmodel.py')

    os.system('python encodeData.py')

    os.system('python buildPredModels.py')

    file = open(MODEL_RESULTS, "r")
    result = file.read()
    file.close()

    print("Model result for current config is: {}".format(result))





