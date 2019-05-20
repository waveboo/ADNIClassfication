from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from keras.models import load_model
import nibabel as nib
import numpy as np
import csv

TEST_CSV = "/home/lb/ADNI/code/ad_mci_cn/2/test.csv"
TRAIN_DATA = "/home/lb/ADNI/data/TRAIN/"

# Read csv
def read_csv(cfile):
    csv_file = open(cfile)
    csv_reader_lines = csv.reader(csv_file)
    raw_data = []
    for one_line in csv_reader_lines:
        raw_data.append(one_line)
    np_data = np.array(raw_data)
    X = np_data[:, 0]
    Y = np_data[:, 1]
    Y = Y.astype(int)
    csv_file.close()
    return X, Y


# Get data from the path to a matrix data
def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data


def get_Y(model_name):
    testX, testY = read_csv(TEST_CSV)
    predY = []
    score = []
    model = load_model(model_name)
    for i, r in enumerate(testX):
        data = read_data(TRAIN_DATA + r)
        data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2], 1)
        p = model.predict(data)
        tmp = np.argmax(np.array(p[0]))
        predY.append(tmp)
    predY = np.array(predY)
    score = np.array(score)
    return testY, predY


if __name__ == '__main__':
    testY, predY = get_Y('/home/lb/ADNI/code/ad_mci_cn/2/weights20.10.hdf5')
    print(accuracy_score(testY, predY))
    w = np.ones(testY.shape[0])
    for idx, j in enumerate(np.bincount(testY)):
        w[testY == idx] *= (j / float(testY.shape[0]))
    print(accuracy_score(testY, predY, sample_weight=w))
    target_names = ['cn', 'mci', 'ad']
    print(classification_report(testY, predY, target_names = target_names))
    print(confusion_matrix(testY, predY))  
