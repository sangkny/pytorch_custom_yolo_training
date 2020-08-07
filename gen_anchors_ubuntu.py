#-*-coding:utf-8-*-
'''
Created on Feb 20, 2017
@author: jumabek
update by sangkny
<<Edit line 130 for your case>>
'''

from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import random 
import math

width_in_cfg_file = 416.
height_in_cfg_file = 416.

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w', encoding="utf-8")

    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32.
        anchors[i][1]*=height_in_cfg_file/32.


    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])

    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))

    f.write('%f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X,centroids,eps,anchor_file):

    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)

        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))

        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)

        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))

        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default='\\path\\to\\voc\\filelist\\train.txt',
                        help='path to filelist\n' )
    parser.add_argument('-output_dir', default='generated_anchors/anchors', type = str,
                        help='Output anchor directory\n' )  
    parser.add_argument('-num_clusters', default=0, type= int,
                        help='number of clusters\n')
    parser.add_argument('-num_classes', default=12, type=int,help='number of classes')
    args = parser.parse_args()

    args.filelist = "/workspace/yolo/data/itms/itms_train_20200729_random.txt"
    args.output_dir = "/workspace/yolo/data/itms/generated_anchors_random"
    args.num_clusters = 0 # 9 anchors if 0, 1-10 clusters
    cls_nums = args.num_classes # class number

    prefix = "" # we need this prefix to generate achors in the local place
    print(sys.getdefaultencoding())
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    f = open(args.filelist, 'rt', encoding="utf-8")

    lines = [line.rstrip('\n') for line in f.readlines()]

    annotation_dims = []

    size = np.zeros((1,1,3))
    cls_hist = np.zeros((1,cls_nums))
    for line in lines:
        line = line.replace('images','labels')
        #line = line.replace('img1','labels')
        #line = line.replace('JPEGImages','labels')


        line = line.replace('.jpg','.txt')
        line = line.replace('.png','.txt')
        # sangkny fix
        line = prefix+line
        line = line.encode('utf8')
        current_line_file = line

        print(line)
        f2 = open(line, 'rt', encoding="utf-8")
        for line in f2.readlines():
            line = line.rstrip('\n')
            w,h = line.split(' ')[3:]       # [cls, c_x, c_y, w, h]
            cls = line.split(' ')[0]
            cls_hist[0][int(cls)] = cls_hist[0][int(cls)]+1 # histogram
            if int(cls) == int(cls_nums-1):
                print('11_etc class: {}'.format(current_line_file))
                print(line)

            #print("------------class:{},  w:{}, h:{}---------".format(cls, w,h))
            annotation_dims.append(tuple(map(float,(w,h))))
    annotation_dims = np.array(annotation_dims)
    print('saving number of classes:', cls_hist)
    cls_hist_file = join(args.output_dir, 'cls_hist_%d.txt'%(cls_nums))
    clsf = open(cls_hist_file, 'w', encoding="utf-8")
    for i in range(cls_nums):
        clsf.write('class {}: {}\n'.format(i, cls_hist[0][i]))

    eps = 0.005

    if args.num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( args.output_dir,'anchors%d.txt'%(num_clusters))

            indices = [random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)
