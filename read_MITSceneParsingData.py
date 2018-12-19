

# coding=utf-8
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir, data_name):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)

    if not os.path.exists(pickle_filepath):  # 不存在文件
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)  # 不存在文件 则下载
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]  # ADEChallengeData2016
        result = create_image_lists(os.path.join(data_dir, data_name))
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:  # 打开pickle文件
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


'''
  返回一个字典:
  image_list{ 
           "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
           "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
           }
'''


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:  # 训练集和验证集 分别制作
        file_list = []
        image_list[directory] = []

        # 获取images目录下所有的图片名
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))  # 加入文件列表  包含所有图片文件全路径+文件名字  如 Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/hi.jpg

        if not file_list:
            print('No files found')
        else:
            for f in file_list:  # 扫描文件列表   这里f对应文件全路径
                # 注意注意，下面的分割符号，在window上为：\\,在Linux撒花姑娘为 : /
                filename = os.path.splitext(f.split("\\")[-1])[0]  # 图片名前缀
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}#  image:图片全路径， annotation:标签全路径， filename:图片名字
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])  # 对图片列表进行洗牌
        no_of_images = len(image_list[directory])  # 包含图片文件的个数
        print('No. of %s files: %d' % (directory, no_of_images))

    return image_list


