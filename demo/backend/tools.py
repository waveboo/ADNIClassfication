import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import cv2
import math


# 定义函数完成文件或文件夹的创建
def mkdir_file(dir_name):
    # 如果不存在文件夹，创建文件
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, 755)
    else:
        # 如果存在文件夹，遍历文件夹中的图片，重复名字进行替换（若可以
        # 存在多张图片，建议用时间戳功能区分，相同名称存取，可能会报错）
        for filename in os.listdir(dir_name):
            if os.path.isfile(os.path.join(dir_name, filename)):
                os.remove(os.path.join(dir_name, filename))


def save_img(path):
    image_data = nib.load(path).get_data()
    dim = image_data.shape
    d1 = dim[0] // 2
    d2 = dim[1] // 2
    d3 = dim[2] // 2
    im1 = np.array(image_data[d1, :, :])
    im1 = np.reshape(im1, (im1.shape[0], im1.shape[1]))
    im2 = np.array(image_data[:, d2, :])
    im2 = np.reshape(im2, (im2.shape[0], im2.shape[1]))
    im3 = np.array(image_data[:, :, d3])
    im3 = np.reshape(im3, (im3.shape[0], im3.shape[1]))
    # im1 = np.flipud(im1.T)
    im2 = np.flipud(im2.T)
    im3 = np.flipud(im3.T)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(im1)
    plt.savefig('1.png')
    plt.close()

    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(im2)
    plt.savefig('2.png')
    plt.close()

    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(im3)
    plt.savefig('3.png')
    plt.close()


def read_info(id):
    dd = dict()
    csv_file = open('csv/out.csv')
    csv_reader_lines = csv.reader(csv_file)
    raw_data = []
    for one_line in csv_reader_lines:
        raw_data.append(one_line)
    np_data = np.array(raw_data)
    csv_file.close()
    for i in np_data:
        if id == i[1]:
            dd['id'] = i[1]
            dd['patient'] = i[2]
            dd['class'] = i[3]
            dd['sex'] = i[4]
            dd['age'] = i[5]
            dd['visit'] = i[6]
            dd['dtype'] = i[7]
            dd['dinfo'] = i[8]
            dd['tp'] = i[9]
            dd['date'] = i[10]
            dd['format'] = i[11]
            return dd


def build_data(dd, type, p):
    if type != "4":
        dd['score'] = p
        if type == "1":
            if 1 - p > 0.5:
                dd['type'] = 'CN'
            else:
                dd['type'] = 'AD'
        elif type == "2":
            if 1 - p > 0.5:
                dd['type'] = 'MCI'
            else:
                dd['type'] = 'AD'
        else:
            if 1 - p > 0.5:
                dd['type'] = 'CN'
            else:
                dd['type'] = 'MCI'
    else:
        tt = np.argmax(p[0])
        dd['score'] = p[0][tt]
        if tt == 0:
            dd['type'] = 'CN'
        elif tt == 1:
            dd['type'] = 'MCI'
        else:
            dd['type'] = 'AD'
    return dd


if __name__ == '__main__':
    save_img(1)
