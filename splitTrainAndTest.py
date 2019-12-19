import random
import os
import subprocess
import sys


def split_data_set(image_dir):
    use_target_dir = True
    use_in_windows = False

    f_val = open("C:/Users/mmc/workspace/yolo/data/itms/itms_test.txt", 'w')
    f_train = open("C:/Users/mmc/workspace/yolo/data/itms/itms_train.txt", 'w')

    target_dir = "/workspace/yolo/data/itms/images"
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
                        f_val.write(target_dir + '/' + f + '\n')  # windows for unix '\r'
                    else:
                        f_val.write(target_dir + '/' + f + '\r')  # windows for unix '\r'
                else:
                    if(use_in_windows):
                        f_val.write(image_dir + '/' + f + '\n') # windows for unix '\r'
                    else:
                        f_val.write(image_dir + '/' + f + '\r')  # windows for unix '\r'
            else:
                if(use_target_dir):
                    if(use_in_windows):
                        f_train.write(target_dir + '/' + f + '\n')
                    else:
                        f_train.write(target_dir + '/' + f + '\r')
                else:
                    if(use_in_windows):
                        f_train.write(image_dir + '/' + f + '\n')
                    else:
                        f_train.write(image_dir + '/' + f + '\r')


split_data_set(sys.argv[1])