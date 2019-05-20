# -*- coding:utf8 -*-
import numpy as np
from random import randint, uniform
import csv
import nibabel as nib
from skimage import transform
from sklearn.model_selection import StratifiedKFold
from ResNetBuild import Resnet3DBuilder
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from keras.optimizers import SGD
from keras import backend as K


# Set global var
def set_globals():
    global RAW_CSV
    global TV_CSV
    global TEST_CSV
    global CSV_PATH
    global TRAIN_DATA

    RAW_CSV = "/home/lb/ADNI/data/CSV/raw.csv"
    TV_CSV = "/home/lb/ADNI/data/CSV/train_val.csv"
    TEST_CSV = "/home/lb/ADNI/data/CSV/test.csv"
    CSV_PATH = "/home/lb/ADNI/data/CSV/"
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


# Split test datas, type 1 = AD/MCI, 2 = AD/CN, 3 = MCI/NC, 4 = AD/MCI/CN
def split_test(type = 4):
    test_sfolder = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)
    Xraw, Yraw = read_csv(RAW_CSV)
    
    if type == 4:
        pass
    else:
        Xtmp = []
        Ytmp = []
        t1 = 0
        t2 = 0
        if type == 3:
            t1 = 1
            t2 = 0
        elif type == 2:
            t1 = 2
            t2 = 0
        elif type == 1:
            t1 = 2
            t2 = 1

        for i in range(len(Xraw)):
            if Yraw[i] == t1 or Yraw[i] == t2:
                Xtmp.append(Xraw[i])
                Ytmp.append(Yraw[i])
        Xraw = np.array(Xtmp)
        Yraw = np.array(Ytmp)

    for tv, test in test_sfolder.split(Xraw, Yraw):
        write_csv(TV_CSV, TEST_CSV, tv, test, Xraw, Yraw)
        break


# Write csv files
def write_csv(a_csv, b_csv, a, b, araw, braw):
    a_out = open(a_csv, 'a', newline='')
    b_out = open(b_csv, 'a', newline='')
    a_writer = csv.writer(a_out)
    b_writer = csv.writer(b_out)
    
    for i in range(len(a)):
        a_writer.writerow([araw[a][i], braw[a][i]])
    for j in range(len(b)):
        b_writer.writerow([araw[b][j], braw[b][j]])
        
    a_out.close()
    b_out.close()


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
def generate_batch_data_random(cc, batch_size, dpath, class_num, is_val):
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
        train_x = np.array([])
        dim = None
        for r in train_x_url:
            single_path = dpath + r
            single_data = read_data(single_path)
            if not is_val:
                single_data = random_flip(single_data)
                single_data = random_crop(single_data)
            dim = single_data.shape
            train_x = np.append(train_x, single_data)
        train_x = train_x.reshape(-1, dim[0], dim[1], dim[2], 1)
        train_y = np.array([])
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
        yield (train_x, train_y)


# Cross_val train models
def train_models(batches = 4, is_cross_val = False, class_num = 3):
    tv_sfolder = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    Xtv, Ytv = read_csv(TV_CSV)
    cross_val = 0

    for train, val in tv_sfolder.split(Xtv, Ytv):
        tpath = CSV_PATH + 'train' + str(cross_val) + '.csv'
        vpath = CSV_PATH + 'val' + str(cross_val) + '.csv'
        write_csv(tpath, vpath, train, val, Xtv, Ytv)

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

        n = class_num
        if class_num == 2:
            n = 1

        model = Resnet3DBuilder.build_resnet_101((128, 128, 128, 1), n, 1e-3)
        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        if class_num == 2:
            model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=sgd)
        else:
            model.compile(loss="categorical_crossentropy", metrics=['acc'], optimizer=sgd)
        checkpointer = ModelCheckpoint(filepath="./models/weights" + str(cross_val) + ".{epoch:02d}.hdf5", save_best_only=True, period=5)

        # Scheduler reduce learn rate
        # def scheduler(epoch):
        #     if epoch % 15 == 0 and epoch != 0:
        #         lr = K.get_value(model.optimizer.lr)
        #         K.set_value(model.optimizer.lr, lr * 0.1)
        #         print("lr changed to {}".format(lr * 0.1))
        #     return K.get_value(model.optimizer.lr)

        earlystoper = EarlyStopping(monitor='val_loss', patience=20)
        tensorboard = TensorBoard(log_dir='./logs', batch_size=batches)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, min_lr=1e-8)
        # reduce_lr = LearningRateScheduler(scheduler)
        model.fit_generator(generate_batch_data_random(aa, batches, TRAIN_DATA, class_num, False),
                            steps_per_epoch=(Xt.size // batches), epochs=500, verbose=1, 
                            validation_data=generate_batch_data_random(bb, batches, TRAIN_DATA, class_num, True),
                            validation_steps=(Xv.size // batches),
                            callbacks=[checkpointer, earlystoper, tensorboard, reduce_lr])
        model.save("./models/weights" + str(cross_val) + ".h5")
        cross_val = cross_val + 1
        if not is_cross_val:
            break


if __name__ == '__main__':
    set_globals()
    split_test(type = 3)
    train_models(batches = 4, class_num = 2)


