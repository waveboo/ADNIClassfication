import os
import subprocess
import time, threading


def file_name(file_dir, ext='nii'):
    L = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.' + ext:
                L.append(os.path.join(dirpath, file))
    return L


def processing_thread(file_names, subjectDir, outputDir, thread_num, total_type_thread_num):
    with open('err.log', 'w') as f:
        f.write('no error recently\n')

    for i, name in enumerate(file_names):
        if i % total_type_thread_num != thread_num:
            continue

        filepath, singlefilename = os.path.split(name)
        print('thread ' + str(thread_num) + ' dealing sample ' + str(i))

        p = subprocess.Popen('recon-all -sd ' + subjectDir + ' -i ' + name + ' -s bert-' + str(thread_num) + '  -autorecon1', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        try:
            out, err = p.communicate(timeout=3000)
            print(len(out), len(err))
            print(err)
        except Exception as e:
            print(e)
            with open('err.log', 'a') as f:
                f.write('err when exc c1: ' + name + '\n')
            t = subprocess.Popen('rm -rf  ' + subjectDir + 'bert-' + str(thread_num), stderr=subprocess.PIPE, shell=True)
            t.wait()
            continue

        if len(err) is not 0:
            with open('err.log', 'a') as f:
                f.write('err c1 retrun err: ' + name + '\n')
            t = subprocess.Popen('rm -rf  ' + subjectDir + 'bert-' + str(thread_num), stderr=subprocess.PIPE, shell=True)
            t.wait()
            continue
        else:
            p = subprocess.Popen('mri_convert ' + subjectDir + 'bert-' + str(thread_num) + '/mri/brainmask.mgz  ' + outputDir + '/' + singlefilename,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            out, err = p.communicate()
            pass

            if len(err) is not 0:
                with open('err.log', 'a') as f:
                    f.write('err c2 retrun err: ' + name + '\n')
                t = subprocess.Popen('rm -rf  ' + subjectDir + 'bert-' + str(thread_num), stderr=subprocess.PIPE, shell=True)
                t.wait()
                continue
            t = subprocess.Popen('rm -rf  ' + subjectDir + 'bert-' + str(thread_num), stderr=subprocess.PIPE, shell=True)
            t.wait()


def freesurfer_preprocessing():
    subjectDir = '/home/lb/ADNI/data/MIX/smri_raw/'
    tmpDir = '/home/lb/ADNI/data/MIX/smri_tmp/'
    outputDir = '/home/lb/ADNI/data/MIX/smri_output/'

    file_names = file_name(subjectDir)
    threads = []

    thread_num = 28
    
    for i in range(thread_num):
        threads.append(threading.Thread(target=processing_thread, args=(file_names, tmpDir, outputDir, i, thread_num)))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    freesurfer_preprocessing()


