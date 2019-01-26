# -*- coding:utf-8 -*-
# ! /usr/bin/env python
"""
@auth:kaaokou
"""
import os
import shutil
import random
import sys
import time
import json

from progressbar import ProgressBar


IMAGE_FORMAT_LIST = ['jpg', 'png', 'gif']
RANDOM_LEVEL = 0.1  # 划分测试级的等级，1/14，数值越大测试集合越小
LABELS_2_NUMBER = 'labels_2_number.json'
NUMBER_2_LABELS= 'number_2_labels.json'
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


def write_dict(dict_obj, path):
    """
    将字典对象以json的形式保存到本地
    """
    content = json.dumps(dict_obj)
    with open(path, 'w') as f:
        f.write(content)


def check_json():
    """
    检测是否存在labels_2_number.json，number_2_labels.json文件
    :return: 返回labels_2_number, number_2_labels
    """
    labels_2_number, number_2_labels = {}, {}
    if os.path.exists(LABELS_2_NUMBER):
        labels_2_number = json.loads(open(LABELS_2_NUMBER).read())
    if os.path.exists(NUMBER_2_LABELS):
        number_2_labels = json.loads(open(NUMBER_2_LABELS).read())
    i = len(labels_2_number) if labels_2_number else 0

    return labels_2_number, number_2_labels, i


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
    test_path = os.path.join(train_root, TEST)
    print('[INFO]:正在划分数据...')
    labels_2_number, number_2_labels, i = check_json()
    progress = ProgressBar()
    for label in progress(os.listdir(train_path)):
        label_dir = os.path.join(train_path, label)
        if not os.path.isdir(label_dir):
            continue
        if labels_2_number.get(label) is None:
            labels_2_number[label] = i
            number_2_labels[i] = label
            i += 1
        train_images = [image for image in os.listdir(label_dir) if image.split('.')[-1].lower()=='jpg']
        # split validation image
        split_images(train_images, label_dir, validation_path, RANDOM_LEVEL)
        split_images(train_images, label_dir, test_path, RANDOM_LEVEL)

    # save json file
    print(labels_2_number, number_2_labels)
    write_dict(labels_2_number, LABELS_2_NUMBER)
    write_dict(number_2_labels, NUMBER_2_LABELS)


def main():
    if len(sys.argv) <= 1:
        print('[ERROR]:请指定需要处理图片文件的源路径...')
        return
    train_path = sys.argv[1]
    split_data(train_path)


if __name__ == '__main__':
    main()

