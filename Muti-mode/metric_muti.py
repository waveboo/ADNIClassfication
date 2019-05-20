from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from keras.models import load_model
import nibabel as nib
import numpy as np
import csv

TEST_CSV = "/home/lb/ADNI/mutimodel/code/mci_cn_later/val1.csv"
FMRI_DATA = "/home/lb/ADNI/mutimodel/data/fmri_later/"
SMRI_DATA = "/home/lb/ADNI/mutimodel/data/smri_tmp/"

# Read csv
def read_csv(cfile):
    csv_file = open(cfile)
    csv_reader_lines = csv.reader(csv_file)
    raw_data = []
    for one_line in csv_reader_lines:
        raw_data.append(one_line)
    np_data = np.array(raw_data)
    data = {}
    for i in np_data:
        if str(i[1]) not in data:
            data[str(i[1])] = {}
            data[str(i[1])][i[0]] = i
        else:
            data[str(i[1])][i[0]] = i
    csv_file.close()
    return data


# Get data from the path to a matrix data
def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data


# type 1 = AD/MCI, 2 = AD/CN, 3 = MCI/NC, 4 = AD/MCI/CN
def code_XY(type, data):
    Xtv = []
    Ytv = []
    cn_num = 0
    mci_num = 1
    ad_num = 2
    if type == 1:
        cn_num = None
        mci_num = 0
        ad_num = 1
    elif type == 2:
        mci_num = None
        ad_num = 1
    elif type == 3:
        ad_num = None
    else:
        pass

    for i in data:
        if data[i]['MRI'][4] == 'CN' and cn_num is not None:
            Xtv.append(i)
            Ytv.append(cn_num)
        elif data[i]['MRI'][4] == 'MCI' and mci_num is not None:
            Xtv.append(i)
            Ytv.append(mci_num)
        elif data[i]['MRI'][4] == 'AD' and ad_num is not None:
            Xtv.append(i)
            Ytv.append(ad_num)
        else:
            pass
    
    Xtv = np.array(Xtv)
    Ytv = np.array(Ytv)

    return Xtv, Ytv


def get_Y(model_name):
    type = 4
    dd = read_csv(TEST_CSV)
    Xtv, Ytv = code_XY(type, dd)
    predY = []
    model = load_model(model_name)
    for i, r in enumerate(Xtv):
        sdata = read_data(SMRI_DATA + dd[r]['MRI'][3] + '.nii')
        sdata = sdata.reshape(1, sdata.shape[0], sdata.shape[1], sdata.shape[2], 1)
        fdata = read_data(FMRI_DATA + dd[r]['fMRI'][3] + 'F.nii')
        fdata = fdata.reshape(1, fdata.shape[0], fdata.shape[1], fdata.shape[2], 1)
        p = model.predict({'input_3':sdata, 'input_4':fdata})
        if p > 0.5:
            tmp = 1
        else:
            tmp = 0
        predY.append(tmp)
    predY = np.array(predY)
    return Ytv, predY


if __name__ == '__main__':
    testY, predY = get_Y('/home/lb/ADNI/mutimodel/code/mci_cn_later/weights1.h5')
    swap = []
    for i in testY:
        if i == max(testY):
            swap.append(1)
        else:
            swap.append(0)
    testY = np.array(swap)
    print(accuracy_score(testY, predY))
    w = np.ones(testY.shape[0])
    for idx, j in enumerate(np.bincount(testY)):
        w[testY == idx] *= (j / float(testY.shape[0]))
    print(accuracy_score(testY, predY, sample_weight=w))
    target_names = ['cn', 'ad']
    print(classification_report(testY, predY, target_names = target_names))
    print(confusion_matrix(testY, predY))  
