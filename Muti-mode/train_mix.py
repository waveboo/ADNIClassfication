# -*- coding:utf8 -*-
import numpy as np
from random import randint, uniform
import csv
import nibabel as nib
from skimage import transform
from sklearn.model_selection import StratifiedKFold
from ResNetBuild import Resnet3DBuilder
from TwoSteamModel import Model2Steam
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from keras.optimizers import SGD
from keras import backend as K
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# Set global var
def set_globals():
    global RAW_CSV
    global CSV_PATH
    global FMRI_DATA
    global SMRI_DATA

    RAW_CSV = "/home/ilab/lb/csv/csv_later/out.csv"
    CSV_PATH = "/home/ilab/lb/csv/csv_later/"
    FMRI_DATA = "/home/ilab/lb/data/fmri_later/"
    SMRI_DATA = "/home/ilab/lb/data/smri_tmp/"


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


# Write csv files
def write_csv(a_csv, b_csv, a, b, araw, braw, data):
    a_out = open(a_csv, 'a', newline='')
    b_out = open(b_csv, 'a', newline='')
    a_writer = csv.writer(a_out)
    b_writer = csv.writer(b_out)

    for i in range(len(a)):
        a_writer.writerow(data[araw[a][i]]['MRI'])
        a_writer.writerow(data[araw[a][i]]['fMRI'])
    for j in range(len(b)):
        b_writer.writerow(data[araw[b][j]]['MRI'])
        b_writer.writerow(data[araw[b][j]]['fMRI'])

    a_out.close()
    b_out.close()


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


def random_flip(single_data, dim=3):
    if dim == 3:
        fliptype = randint(0, 7)
        if fliptype == 0:
            single_data = np.flip(single_data, 0)
        elif fliptype == 1:
            single_data = np.flip(single_data, 1)
        elif fliptype == 2:
            single_data = np.flip(single_data, 2)
        elif fliptype == 3:
            single_data = np.flip(single_data, 0)
            single_data = np.flip(single_data, 1)
        elif fliptype == 4:
            single_data = np.flip(single_data, 0)
            single_data = np.flip(single_data, 2)
        elif fliptype == 5:
            single_data = np.flip(single_data, 1)
            single_data = np.flip(single_data, 2)
        elif fliptype == 6:
            single_data = np.flip(single_data, 0)
            single_data = np.flip(single_data, 1)
            single_data = np.flip(single_data, 2)
        else:
            pass
    elif dim == 2:
        fliptype = randint(0, 3)
        if fliptype == 0:
            single_data = np.flip(single_data, 0)
        elif fliptype == 1:
            single_data = np.flip(single_data, 1)
        elif fliptype == 2:
            single_data = np.flip(single_data, 0)
            single_data = np.flip(single_data, 1)
        else:
            pass
    else:
        pass
    return single_data


def random_crop(single_data, dim=3):
    if dim == 3:
        d = single_data.shape
        is_crop = randint(0, 1)
        if is_crop == 0:
            pass
        else:
            crop_rate = uniform(0.8, 1)
            start_rate = uniform(0, 0.2)
            crop_len = [int(crop_rate * d[0]), int(crop_rate * d[1]), int(crop_rate * d[2])]
            start_point = [int(start_rate * d[0]), int(start_rate * d[1]), int(start_rate * d[2])]
            single_data_crop = single_data[start_point[0]:(start_point[0] + crop_len[0]),
                               start_point[1]:(start_point[1] + crop_len[1]),
                               start_point[2]:(start_point[2] + crop_len[2])]
            single_data = transform.resize(single_data_crop, (d[0], d[1], d[2]))
    elif dim == 2:
        d = single_data.shape
        is_crop = randint(0, 1)
        if is_crop == 0:
            pass
        else:
            crop_rate = uniform(0.8, 1)
            start_rate = uniform(0, 0.2)
            crop_len = [int(crop_rate * d[0]), int(crop_rate * d[1])]
            start_point = [int(start_rate * d[0]), int(start_rate * d[1])]
            single_data_crop = single_data[start_point[0]:(start_point[0] + crop_len[0]),
                               start_point[1]:(start_point[1] + crop_len[1]), :]
            single_data = transform.resize(single_data_crop, (d[0], d[1], d[2]))
    else:
        pass
    return single_data

# Generate batch data to same memory
def generate_batch_data_random(cc, data, batch_size, is_val, type, key):
    ylen = len(cc)
    loopcount = ylen // batch_size
    np.random.shuffle(cc)
    x = np.array([])
    y = np.array([])
    for i in range(0, ylen):
        x = np.append(x, cc[i]['x'])
        y = np.append(y, cc[i]['y'])
    while (True):
        seed = randint(0, loopcount - 1)
        train_x_url = x[seed * batch_size:(seed + 1) * batch_size]
        train_smri = np.array([])
        train_fmri = np.array([])
        sdim = None
        fdim = None
        for r in train_x_url:
            smri_path = SMRI_DATA + data[r]['MRI'][3] + '.nii'
            smri_data = read_data(smri_path)
            if not is_val:
                smri_data = random_flip(smri_data, 3)
                smri_data = random_crop(smri_data, 3)
            sdim = smri_data.shape
            train_smri = np.append(train_smri, smri_data)

            fmri_path = FMRI_DATA + data[r]['fMRI'][3] + 'F.nii'
            fmri_data = read_data(fmri_path)
            if not is_val:
                fmri_data = random_flip(fmri_data, 2)
                fmri_data = random_crop(fmri_data, 2)
            fdim = fmri_data.shape
            train_fmri = np.append(train_fmri, fmri_data)

        train_smri = train_smri.reshape(-1, sdim[0], sdim[1], sdim[2], 1)
        train_fmri = train_fmri.reshape(-1, fdim[0], fdim[1], fdim[2], 1)

        train_y = np.array([])
        class_num = 2
        if type == 4:
            class_num = 3
        for b in range(seed * batch_size, (seed + 1) * batch_size):
            tmp = [0 for i in range(class_num)]
            if class_num > 2:
                tmp[int(y[b])] = 1
            else:
                tmp = 0
                if int(y[b]) == int(max(y)):
                    tmp = 1
            train_y = np.append(train_y, tmp)
        if class_num > 2:
            train_y = train_y.reshape(-1, class_num)
        else:
            pass
        key1 = 'input_' + str(2 * key - 1)
        key2 = 'input_' + str(2 * key)
        yield ({key1:train_smri, key2:train_fmri}, train_y)


# Cross_val train models
def train_models(batches = 4, is_cross_val = False, type = 4):
    tv_sfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    dd = read_csv(RAW_CSV)
    Xtv, Ytv = code_XY(type, dd)

    cross_val = 0
    for train, val in tv_sfolder.split(Xtv, Ytv):
        tpath = CSV_PATH + 'train' + str(cross_val) + '.csv'
        vpath = CSV_PATH + 'val' + str(cross_val) + '.csv'
        write_csv(tpath, vpath, train, val, Xtv, Ytv, dd)

        Xt = Xtv[train]
        Yt = Ytv[train]
        Xv = Xtv[val]
        Yv = Ytv[val]
        aa = np.array([])
        for a in range(0, Xt.size):
            aa = np.append(aa, {'x': Xt[a], 'y': Yt[a]})
        bb = np.array([])
        for b in range(0, Xv.size):
            bb = np.append(bb, {'x': Xv[b], 'y': Yv[b]})
        
        n = 1
        if type == 4:
            n = 3

        model = Model2Steam.build_model_lenet((64, 64, 64, 1), (64, 64, 64, 1), n, 1e-4)

        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        if type == 4:
            model.compile(loss="categorical_crossentropy", metrics=['acc'], optimizer=sgd)
        else:
            model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=sgd)

        checkpointer = ModelCheckpoint(filepath="./models/weights" + str(cross_val) + ".{epoch:02d}.hdf5", save_best_only=True, period=5)
        earlystoper = EarlyStopping(monitor='val_loss', patience=20)
        tensorboard = TensorBoard(log_dir='./logs', batch_size=batches)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=6, verbose=1, min_lr=1e-8)
        # reduce_lr = LearningRateScheduler(scheduler)
        model.fit_generator(generate_batch_data_random(aa, dd, batches, False, type, cross_val + 1),
                            steps_per_epoch=4*(Xt.size // batches), epochs=200, verbose=1,
                            validation_data=generate_batch_data_random(bb, dd, batches, True, type, cross_val + 1),
                            validation_steps=(Xv.size // batches),
                            callbacks=[checkpointer, earlystoper, tensorboard, reduce_lr])
        model.save("./models/weights" + str(cross_val) + ".h5")
        cross_val = cross_val + 1
        if not is_cross_val:
            break


if __name__ == '__main__':
    set_globals()
    train_models(batches = 8, is_cross_val = True, type = 3)

