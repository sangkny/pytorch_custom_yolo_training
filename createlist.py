import glob
import os
import numpy as np
import sys

current_dir = "./data/snowman/images" # all the images before splitting
#target_dir = "/workspace/yolo/data/snowman/images" # for ubuntu training
target_dir = "./data/snowman/images"
split_pct = 10;
file_train = open("data/snowman/train.txt", "w")
file_val = open("data/snowman/val.txt", "w")
counter = 1  
index_test = round(100 / split_pct)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(target_dir + "/" + title + '.jpg' + "\n")
        else:
                file_train.write(target_dir + "/" + title + '.jpg' + "\n")
                counter = counter + 1
file_train.close()
file_val.close()
