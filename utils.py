# coding=utf-8
"""
@author: kaaokou
"""
import os
import json
import logging

import tensorflow as tf
import pandas as pd


def get_labels_2_number(path='labels_2_number.json'):
    """
    get labels_2_number
    :return:dict
    """
    return json.loads(open(path).read())


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    归一化图像数据
    """
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)


def get_images(path):
    """
    得到images文件路径列表， 只能够获取jpg格式
    """
    ret = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1].lower() == 'jpg':
                ret.append(os.path.join(root, file))
    return ret


def read_label_df(path):
    """
    读取labels.csv的dataframe
    """
    label_df = pd.read_csv(path, index_col=0, names=["label"])
    return label_df


def train_log(filename='logfile'):
    # create logger
    logger_name = "filename"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    log_path = './' + filename + '.log'
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    # create formatter
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger