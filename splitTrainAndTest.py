import random
import os
#import subprocess
import argparse
#import sys

parser = argparse.ArgumentParser(description='Spit data into train and test lists for YOLOv3')
parser.add_argument('--image_dir', help='Path to image files.')
parser.add_argument('--target_dir', type=str, default='/workspace/yolo/data/itms/images', help='relative path for ubuntu training')
parser.add_argument('--use_target_dir', type=int, default=1, help='Flag to use target dir instead of using img_dir')
parser.add_argument('--use_in_windows', type=int, default=0, help='Flag to use it in windows')
args = parser.parse_args()

# temp parameter settings
args.image_dir = "C:/Users/mmc/workspace/yolo/data/itms/images"     # absolute path which contains origial images
args.target_dir = "/workspace/yolo/data/itms/images"                # relative path in ubuntu
args.use_target_dir = True
args.use_in_windows = False

def split_data_set(arg):
    use_target_dir = arg.use_target_dir
    use_in_windows = arg.use_in_windows
    f_val = open("C:/Users/mmc/workspace/yolo/data/itms/itms_val_20200117.txt", 'w')
    f_train = open("C:/Users/mmc/workspace/yolo/data/itms/itms_train_20200117.txt", 'w')
    image_dir = arg.image_dir
    target_dir = arg.target_dir # ""
    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    data_test_size = int(0.1 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)

    for f in os.listdir(image_dir):
        if (f.split(".")[1].lower() == "jpg"):
            ind += 1

            if ind in test_array:
                if(use_target_dir):
                    if(use_in_windows):
                        f_val.write(target_dir + '/' + f + '\r\n')  # windows for unix '\n'
                    else:
                        f_val.write(target_dir + '/' + f + '\n')  # windows for unix '\n'
                else:
                    if(use_in_windows):
                        f_val.write(image_dir + '/' + f + '\r\n') # windows for unix '\r'
                    else:
                        f_val.write(image_dir + '/' + f + '\n')  # windows for unix '\n'
            else:
                if(use_target_dir):
                    if(use_in_windows):
                        f_train.write(target_dir + '/' + f + '\r\n')
                    else:
                        f_train.write(target_dir + '/' + f + '\n')
                else:
                    if(use_in_windows):
                        f_train.write(image_dir + '/' + f + '\r\n')
                    else:
                        f_train.write(image_dir + '/' + f + '\n')  # \r only for mac
    f_val.close()
    f_train.close()


split_data_set(args)