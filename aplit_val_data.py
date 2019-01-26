#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:16:22 2019

@author: shinong
"""

import os
import shutil
import random
import sys
import time

from progressbar import ProgressBar

IMAGE_FORMAT_LIST = ['jpg', 'png', 'gif']
RANDOM_LEVEL = 0.3  # 划分测试级的等级，1/14，数值越大测试集合越小
VALIDATION = 'validation'
TEST = 'test'

def decorate(func):
    def call_back(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print('[INFO]:此次操作耗时<{:.2f}>秒'.format(end - start))
        return ret

    return call_back

def check_dir_exits(path):
    """
    检查需要的文件夹是否存在，不存在则创建
    """
    if not os.path.exists(path):
        os.makedirs(path)

def split_images(train_images, label_dir, split_path,random_level):
    """
    划分数据
    :param train_images:需要划分的image
    :param label_dir:标签文件夹
    :param split_path:目的路径
    :param random_level:划分的百分比，0-1的浮点数
    """
    images = random.sample(train_images, int(random_level * len(train_images)))
    label = label_dir.split('/')[-1]
    for image in images:
        # mv image --> validation_path
        dst_dir = os.path.join(split_path, label)
        check_dir_exits(dst_dir)
        dst_path = os.path.join(dst_dir, image)
        shutil.move(os.path.join(label_dir, image), dst_path)
        train_images.remove(image)
        
@decorate
def split_data(train_path='data/train'):
    """
    划分验证集，和测试集，只支持jpg格式图像
    """
    train_root = os.path.dirname(train_path)
    validation_path = os.path.join(train_root, VALIDATION)
    print('[INFO]:正在划分数据...')
    progress = ProgressBar()
    for label in progress(os.listdir(train_path)):
        label_dir = os.path.join(train_path, label)
        if not os.path.isdir(label_dir):
            continue
        train_images = [image for image in os.listdir(label_dir) if image.split('.')[-1].lower()=='jpg']
        # split validation image
        split_images(train_images, label_dir, validation_path, RANDOM_LEVEL)


def main():
    if len(sys.argv) <= 1:
        print('[ERROR]:请指定需要处理图片文件的源路径...')
        return
    train_path = sys.argv[1]
    split_data(train_path)


if __name__ == '__main__':
    main()