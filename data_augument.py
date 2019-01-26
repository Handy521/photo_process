# -*- coding:utf-8 -*-
"""
@auth:kaaokou
"""

import os
import random
import sys
import time

import cv2
import numpy as np
from progressbar import ProgressBar
from imgaug import augmenters as iaa

IMAGE_FORMAT_LIST = ['jpg', 'png', 'gif']
MAX_NUM = 1000


def decorate(func):
    def call_back(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[INFO]:此次操作一共耗时<{:.2f}>秒'.format(end - start))

    return call_back


def rotate_image(image, angle):
    """
    输入图片，输出顺时针旋转后的图片
    :param image: 原始图片对象
    :param angle: 旋转角度，例如90
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def crop_image(img):
    """
    随机裁剪图片，25~75%
    """
    h, w = img.shape[:2]
    r = [random.random()/4 for _ in range(4)]
    new_image = img[int(r[0]*h):int((1-r[1])*h), int(r[2]*w):int((1-r[3])*w)]
    return new_image


def get_augmented_image(img_bgr, cnt):
    """
    给定一个bgr格式的图像，输出4个上下左右镜像的图像
    :param img_bgr: opencv image 对象
    :return: [img_bgr*4]
    """
    if random.randint(1, 3) == 1:
        img_bgr = rotate_image(img_bgr, random.randint(1, 3)*90)
    if random.randint(1, 3) == 1:
        img_bgr = crop_image(img_bgr)
    # BGR转RGB，方法2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    array = [np.array(img_rgb)]

    # 变换的规则
    seq = iaa.Sequential([
        iaa.SomeOf(2, [
            # 水平翻转
            iaa.Fliplr(0.5),
            # 垂直翻转
            iaa.Flipud(0.5),
            # 明暗变化
            iaa.Multiply((0.7, 1.3)),
            # 高斯模糊
            iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
            # 添加噪点
            #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
            # 添加dropout
            #iaa.OneOf([
                #iaa.Dropout((0.01, 0.01), per_channel=0.5), # randomly remove up to 10% of the pixels
                #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
            #]),
            # 不用这个旋转
            #iaa.Affine(rotate=(-45, 45))
        ], random_order=True),
    ])

    img_lists = []
    for _ in range(cnt):
        image_rgb = seq.augment_images(array)
        image_brg = cv2.cvtColor(image_rgb[0], cv2.COLOR_RGB2BGR)
        img_lists.append(image_brg)

    return img_lists


@decorate
def augmented_images(root_path='./train', cnt=4):
    """
    增广图像
    """
    dir_path = root_path

    print('[INFO]:正在处理<{}>路径下的图像...'.format(dir_path))
    # 添加进度条显示
    progress = ProgressBar()
    for image in progress(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, image)
        image_data = cv2.imread(image_path)
        if image_data is None:
            continue

        image_lists = get_augmented_image(image_data, cnt)
        for i, new_image in enumerate(image_lists):
            dst_image_path = dir_path  +'/aug'+ str(i) + image
            cv2.imwrite(dst_image_path, new_image)


def del_redundant_image(src_path):
    """
    删除多余的图片
    """
    for path in os.listdir(src_path):
        root_path = os.path.join(src_path, path)
        print('[INFO]:正在处理<{}>...'.format(root_path))
        file_cnt = len(os.listdir(root_path))
        if file_cnt > MAX_NUM:
            file_lists = [file for file in os.listdir(root_path) if file.startswith('augment') ]
            if not file_lists:
                continue
            del_file_lists = random.sample(file_lists, (file_cnt-MAX_NUM))
            for del_file in del_file_lists:
                del_file = os.path.join(root_path, del_file)
                # print('[INFO]:正在删除文件<{}>...'.format(del_file))
                os.remove(del_file)


def main():
#    if len(sys.argv) <= 1:
#        print('[ERROR]:请指定需要增广的文件路径起始路劲...')
#        return
    src_path ='/media/shinong/study/30easy_dl/train'
    for path in os.listdir(src_path):
        root_path = os.path.join(src_path, path)
#        file_cnt = len(os.listdir(root_path))
#        if abs(file_cnt - MAX_NUM) > 300:
#            cnt = MAX_NUM//file_cnt
        augmented_images(root_path, 2)
    # del redundant image
#    del_redundant_image(src_path)


if __name__ == '__main__':
    main()

