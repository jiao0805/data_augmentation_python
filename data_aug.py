# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:47:41 2018

@author: DeepLearning
"""
import math
import numpy as np
from skimage import io
from PIL import Image
from PIL import ImageFilter
from keras.preprocessing import image
import cv2

#add noise into only rgb image
def noise(img_rgb,img_range,counter):
    num_of_noise=10000
    height,weight,channel = img_rgb.shape
    img_rgb_noise = img_rgb
    for i in range(num_of_noise):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img_rgb_noise[x,y,:]=0
    io.imsave('train_augmenta/'+str(counter)+'_noise.jpg',img_rgb_noise)
    io.imsave('train_augmenta/'+str(counter)+'_noise.png',img_range)
    
#flip both rgb and range vertically and horizontally
def flip(img_rgb,img_range,counter):
    #vertical filp for both rgb and range
    img_rgb_flipver = img_rgb[::-1,:,:]
    img_range_flipver = img_range[::-1,:,:]
    io.imsave('train_augmenta/'+str(counter)+'_flipver.jpg',img_rgb_flipver)
    io.imsave('train_augmenta/'+str(counter)+'_flipver.png',img_range_flipver)
    #horizontal flip of both rgb and range
    img_rgb_flipbor = img_rgb[:,::-1,:]
    img_range_flipbor = img_range[:,::-1,:]
    io.imsave('train_augmenta/'+str(counter)+'_flipbor.jpg',img_rgb_flipbor)
    io.imsave('train_augmenta/'+str(counter)+'_flipbor.png',img_range_flipbor)
    
#rotate both rgb and rangy by three random angle with in 90
def rotate(img_rgb,img_range,counter,row_axis=0,col_axis=1,channel_axis=2,fill_mode='wrap',cval=0.):
    rotate_limit_1 = (-80,80)
    for i in range(5):
        theta_1 = np.pi / 180 * np.random.uniform(rotate_limit_1[0], rotate_limit_1[1])
        rotation_matrix = np.array([[np.cos(theta_1), -np.sin(theta_1), 0],
                                 [np.sin(theta_1), np.cos(theta_1), 0],
                                 [0, 0, 1]])
        h, w = img_rgb.shape[row_axis], img_rgb.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
        img_rgb_rotate = image.apply_transform(img_rgb, transform_matrix, channel_axis, fill_mode, cval)
        img_range_rotate = image.apply_transform(img_range, transform_matrix, channel_axis, fill_mode, cval)
        io.imsave('train_augmenta/'+str(counter)+'_rotate_'+str(i)+'.jpg',img_rgb_rotate)
        io.imsave('train_augmenta/'+str(counter)+'_rotate_'+str(i)+'.png',img_range_rotate)

def GaussianBlur(img_rgb_name,img_range,counter):
    img_rgb=Image.open(img_rgb_name)
    img_rgb_blur=img_rgb.filter(ImageFilter.GaussianBlur(radius=2))
    io.imsave('train_augmenta/'+str(counter)+'_blur2.jpg',img_rgb_blur)
    io.imsave('train_augmenta/'+str(counter)+'_blur2.png',img_range)
    img_rgb_blur=img_rgb.filter(ImageFilter.GaussianBlur(radius=3))
    io.imsave('train_augmenta/'+str(counter)+'_blur3.jpg',img_rgb_blur)
    io.imsave('train_augmenta/'+str(counter)+'_blur3.png',img_range)

def RT(img, center=(50, 100)):
     U, V, p = img.shape
     RTimg = img.copy() - img
     m, n = center
     for u in range(U):
         d = 2 * math.pi * u / U
         for v in range(V):
             x = v * math.sin(d)
             y = v * math.cos(d)
             mm = int(math.floor(m + x))
             nn = int(math.floor(n + y))
             if mm>=0 and mm<U and nn>=0 and nn<V:
                 RTimg[u, v, :] = img[mm, nn, :]
     return RTimg


for counter in range(5688):
    img_rgb_name='greytrain/'+str(counter)+'.jpg'
    img_range_name ='greytrain/'+str(counter)+'.png'
    img_rgb = io.imread(img_rgb_name)
    img_range = io.imread(img_range_name)
    print(img_range.shape)
    print(img_rgb.shape)
    io.imsave('train_augmenta/'+str(counter)+'.jpg',img_rgb)
    io.imsave('train_augmenta/'+str(counter)+'.png',img_range)
    print(img_rgb_name)
    print(img_range_name)
    img_range_3=np.array([img_range for i in range(3)])
    img_range_3=img_range_3.transpose(1,2,0)
    img_rgb_3=np.array([img_rgb for i in range(3)])
    img_rgb_3=img_rgb_3.transpose(1,2,0)
    flip(img_rgb_3,img_range_3,counter)
    rotate(img_rgb_3,img_range_3,counter)
    GaussianBlur(img_rgb_name,img_range,counter)
    noise(img_rgb_3,img_range,counter)
    
