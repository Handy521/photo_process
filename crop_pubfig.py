# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:09:14 2019

@author: yong2
将pubfig数据集人脸图片裁剪只保留人脸存到新的路径
"""

import cv2
import os
import glob


def read_txt(path):
    """读取人脸边界框信息，返回框内人脸"""
    with open(path, 'r') as f:
        annotations = f.readlines()
    annotation = annotations[0].strip().split(' ')
    im_path = path[:-3]+'jpg'
    try:
        img = cv2.imread(im_path,cv2.IMREAD_UNCHANGED)
    except:
        print(im_path)
        
    bound=min(img.shape[0:2])
    bbox = list(map(int, annotation)) 
    x1=int(bbox[0]-bound/9)
    y1=int(bbox[1]-bound/9)
    x2=int(bbox[2]+bound/9)
    y2=int(bbox[3]+bound/9)
    if x1<0:x1=0
    if y1<0:y1=0
    if x2>img.shape[1]:x2=img.shape[1]
    if y2>img.shape[0]:y2=img.shape[0]
    new_image=img[y1:y2,x1:x2,:]
    image=cv2.resize(new_image,(160,160),interpolation=cv2.INTER_LINEAR)

    return image

def save_photo(pre_path,new_path):   
    """遍历数据集下图片和人脸框的文本，保存新的图片"""
    sub_dirs=os.listdir(input_data)
    for sub_dir in sub_dirs:    
        file_list=[]
        file_glob=os.path.join(input_data,sub_dir,'*.'+'txt')
        file_list.extend(glob.glob(file_glob))#非常强大，读取当前路径的所有文件
        
        for i in range(len(file_list)):
            path=os.path.join(file_list[i])      
            image=read_txt(path)
            #新建人脸图片路径
            new_path=os.path.join(new_dir,sub_dir)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            save_file = os.path.join(new_path, "%d.jpg"%i) 
            cv2.imwrite(save_file, image)
            
if __name__=='__main__':
    input_data="pubFig/"
    new_dir='PUB_FIG/'
    save_photo(input_data,new_dir)        
        
        
        
        
        
        
        
        