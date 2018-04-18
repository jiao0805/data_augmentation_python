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

#add noise into only rgb image
def noise(img_rgb,img_range):
    num_of_noise=10000
    height,weight,channel = img_rgb.shape
    img_rgb_noise = img_rgb
    for i in range(num_of_noise):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img_rgb_noise[x,y,:]=0
    io.imsave('916_noise.jpg',img_rgb_noise)
    io.imsave('916_noise.png',img_range)
    
#flip both rgb and range vertically and horizontally
def flip(img_rgb,img_range):
    #vertical filp for both rgb and range
    img_rgb_flipver = img_rgb[::-1,:,:]
    img_range_flipver = img_range[::-1]
    io.imsave('916_flipver.jpg',img_rgb_flipver)
    io.imsave('916_flipver.png',img_range_flipver)
    #horizontal flip of both rgb and range
    img_rgb_flipbor = img_rgb[:,::-1,:]
    img_range_flipbor = img_range[:-1:]
    io.imsave('916_flipbor.jpg',img_rgb_flipbor)
    io.imsave('916_flipbor.png',img_range_flipbor)
    
#rotate both rgb and rangy by three random angle with in 90
def rotate(img_rgb,img_range,row_axis=0,col_axis=1,channel_axis=2,fill_mode='wrap',cval=0.):
    rotate_limit_1 = (-60,60)
    for i in range(3):
        theta_1 = np.pi / 180 * np.random.uniform(rotate_limit_1[0], rotate_limit_1[1])
        rotation_matrix = np.array([[np.cos(theta_1), -np.sin(theta_1), 0],
                                 [np.sin(theta_1), np.cos(theta_1), 0],
                                 [0, 0, 1]])
        h, w = img_rgb.shape[row_axis], img_rgb.shape[col_axis]
        transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
        img_rgb_rotate = image.apply_transform(img_rgb, transform_matrix, channel_axis, fill_mode, cval)
        img_range_rotate = image.apply_transform(img_range, transform_matrix, channel_axis, fill_mode, cval)
        io.imsave('916_rotate_'+str(i)+'.jpg',img_rgb_rotate)
        io.imsave('916_rotate_'+str(i)+'.png',img_range_rotate)

def GaussianBlur(img_rgb_name,img_range):
    img_rgb=Image.open(img_rgb_name)
    img_rgb_blur=img_rgb.filter(ImageFilter.GaussianBlur(radius=2))
    io.imsave('916_blur.jpg',img_rgb_blur)
    io.imsave('916_blur.png',img_range)

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
     
     
img_rgb_name='916.jpg'
img_rgb = io.imread('916.jpg')
img_range = io.imread('916.png')
img_test=np.array([img_range for i in range(3)])
img_test=img_test.transpose(1,2,0)
flip(img_rgb,img_range)
noise(img_rgb,img_range)
rotate(img_rgb,img_test)
GaussianBlur(img_rgb_name,img_range)
RT_img=RT(img_rgb)
io.imsave('916_test_.png',RT_img)

