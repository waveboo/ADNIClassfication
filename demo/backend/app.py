from flask import Flask, url_for, request, send_from_directory
from tools import mkdir_file, save_img, read_info, build_data
from metric import metric_single, metric_muti
from flask_cors import *
import os, json
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

global TYPE
global MODEL


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/upload_smri', methods=['POST'])
def upload_smri():
    mode = request.form['mode']
    BASE_DIR = os.path.dirname(__file__)
    mkdir_file(os.path.join(BASE_DIR, 'file/smri/' + str(mode)))
    # 获取前端传输的文件(对象)
    f = request.files.get('file')
    filename = f.filename
    types = ['nii']
    if filename.split('.')[-1] in types:
        f.save(os.path.join(BASE_DIR, 'file/smri/' + str(mode) + '/{0}'.format(filename)))
        save_img('file/smri/' + str(mode) + '/{0}'.format(filename))
        # 返回给前端结果
        return json.dumps(
            {'code': 200, 'filename': filename})
    else:
        return json.dumps({'error': '文件格式不合法', 'code': 400})


@app.route('/upload_fmri', methods=['POST'])
def upload_fmri():
    BASE_DIR = os.path.dirname(__file__)
    mkdir_file(os.path.join(BASE_DIR, 'file/fmri/'))
    f = request.files.get('file')
    filename = f.filename
    types = ['nii']
    if filename.split('.')[-1] in types:
        f.save(os.path.join(BASE_DIR, 'file/fmri/{0}'.format(filename)))
        # 返回给前端结果
        return json.dumps(
            {'code': 200})
    else:
        return json.dumps({'error': '文件格式不合法', 'code': 400})


@app.route('/get_img', methods=['GET'])
def get_img():
    filename = request.args.get("filename")
    BASE_DIR = os.path.dirname(__file__)
    return send_from_directory(BASE_DIR, filename, as_attachment=True)


@app.route('/single_mode', methods=['POST'])
def single_mode():
    filename = request.form.get("filename")
    dd = read_info(filename.split('.')[0])
    type = request.form.get("type")
    # p = metric_single(type, 'file/smri/1/' + filename)
    p = [[0.8966226262, 0.22, 0.1]]
    return json.dumps(build_data(dd, type, p))


@app.route('/muti_mode', methods=['POST'])
def muti_mode():
    filename = request.form.get("filename")
    dd = read_info(filename.split('.')[0])
    type = request.form.get("type")
    # p = metric_muti(type, 'file/smri/2/' + filename)
    p = [[0.8966226262, 0.22, 0.1]]
    return json.dumps(build_data(dd, type, p))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
