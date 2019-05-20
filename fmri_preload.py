# -*- coding:utf8 -*-
import os
import csv
import shutil	
import numpy as np
import nibabel as nib
from skimage import transform
import cv2

fmri_row_dir = '/home/lb/homex/datasets/ADNI-SMRI/MIX/Axial_rsfMRI__Eyes_Open_/fmri/FunImg/'
fmri_from_dir = '/home/lb/homex/datasets/ADNI-SMRI/MIX/Axial_MB_rsfMRI__Eyes_Open_/fmri/FunImgTARWSDCF/'
smri_from_dir = '/home/lb/homex/datasets/ADNI-SMRI/MIX/Axial_rsfMRI__Eyes_Open_/smri/'
csv_from = '/home/lb/homex/datasets/ADNI-SMRI/MIX/Axial_rsfMRI__Eyes_Open_/out.csv'
fmri_tar_dir = '/home/lb/ADNI/data/MIX/fmri_later/'
fmri_rawtar_dir = '/home/lb/ADNI/data/MIX/fmri_raw/'
smri_tar_dir = '/home/lb/ADNI/data/MIX/smri/'
smri_aa_dir = '/home/lb/ADNI/data/MIX/smri_raw/'
csv_tar = '/home/lb/ADNI/data/MIX/out.csv'


# Get data from the path to a matrix data
def read_data(path):
    image_data = nib.load(path).get_data()
    return image_data


def read_matrix(path):
    image_data = nib.load(path).affine
    return image_data


def save_img(data, affine, filename):
    us = nib.Nifti1Image(data, affine)
    nib.save(us, fmri_rawtar_dir + filename)
  

def copy_img(old_file, tar_file):
    shutil.copy(smri_from_dir + old_file, smri_aa_dir + tar_file)


# Read csv
def read_csv(cfile):
    csv_file = open(cfile)
    csv_reader_lines = csv.reader(csv_file)
    raw_data = []
    for one_line in csv_reader_lines:
        raw_data.append(one_line)
    np_data = np.array(raw_data)
    csv_file.close()
    return np_data


# Write csv files
def write_csv(a_csv, ll):
    a_out = open(a_csv, 'a', newline='')
    a_writer = csv.writer(a_out)
    a_writer.writerow(ll)
    a_out.close()


def normalize_fmri(ff):
    img = read_data(ff)
    img_shape = img.shape
    print(img_shape)
    low_bound = int(img_shape[3] / 2) - 32
    img_sub = img[:, :, int(img_shape[2] / 2  + 1), low_bound: low_bound + 64]
    single_data = transform.resize(img_sub, (64, 64, 64))
    aff = read_matrix(ff)
    return single_data, aff


def row_fmri(ff):
    fls = os.listdir(ff)
    low_bound = int(len(fls) / 2) - 32
    aff = read_matrix(ff + fls[0])
    data_shape = read_data(ff + fls[0]).shape
    single_data = []
    for i in fls[low_bound: low_bound + 64]:
        tmp = read_data(ff + i)
        single_data.append(tmp[:, :, int(data_shape[2] / 2)])
    single_data = np.concatenate(single_data, axis = 2)
    single_data = transform.resize(single_data, (64, 64, 64))
    return single_data, aff
    

raw = read_csv(csv_from)
data = {}
for i in raw:
    if str(i[1]) not in data:
        data[str(i[1])] = {}
        data[str(i[1])][i[0]] = i
    else:
        data[str(i[1])][i[0]] = i

for j in data:
    # nor0, aff0 = normalize_fmri(fmri_from_dir + data[j]['fMRI'][3] + '/' + data[j]['fMRI'][3] + 'F.nii')
    nor, aff = row_fmri(fmri_row_dir + data[j]['fMRI'][3] + '/')
    save_img(nor, aff, data[j]['fMRI'][3] + 'F.nii')
    copy_img(data[j]['MRI'][3]+ '/' + os.listdir(smri_from_dir + data[j]['MRI'][3])[0], data[j]['MRI'][3] + '.nii')

