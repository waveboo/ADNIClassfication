from keras.models import load_model
import nibabel as nib
import os
import numpy as np


# Get data from the path to a matrix data
def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data


def metric_single(type, path):
    if type == '1':
        model = load_model('ad_cn')
    elif type == '2':
        model = load_model('ad_mci')
    elif type == '3':
        model = load_model('mci_cn')
    elif type == '4':
        model = load_model('ad_mci_cn')
    else:
        pass
    data = read_data(path)
    data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2], 1)
    p = model.predict(data)
    return p


def metric_muti(type, path):
    if type == '1':
        model = load_model('ad_cn')
    elif type == '2':
        model = load_model('ad_mci')
    elif type == '3':
        model = load_model('mci_cn')
    elif type == '4':
        model = load_model('ad_mci_cn')
    else:
        pass
    sdata = read_data(path)
    sdata = sdata.reshape(1, sdata.shape[0], sdata.shape[1], sdata.shape[2], 1)
    ffile = os.listdir('file/fmri')[0]
    fdata = read_data('file/fmri' + ffile)
    fdata = fdata.reshape(1, fdata.shape[0], fdata.shape[1], fdata.shape[2], 1)
    p = model.predict({'input_1': sdata, 'input_2': fdata})
    return p
