import cv2
import numpy as np
import glob
import random
import pathlib
import subprocess, os
from PIL import Image
# import matplotlib.pyplot as plt

#ui
import cv2
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.ttk import Button, Style, Progressbar
import time
import os

folder_selected=""
class_name=""
data_class=""
v_class=""

def add_class(): 
    data_class = add_class_field.get()
    class_name = data_class
    if data_class == "":
        messagebox.showinfo("Notif", "Class cant be empty")
    else:
        print("===============Class Name================")
        print(class_name)
        v_class=class_name
        main()


# Load Yolo
def main():
    folder_selected = filedialog.askdirectory()
    v_class =add_class_field.get()
    print("=========================class name===================")
    print(v_class)
    print(folder_selected)
    print( "========================================")
    # label_file_explorer.configure(text="File Path: "+folder_selected)
    # sangkny
    # original net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
    modelBaseDir = "C:/Users/mmc/workspace/yolo"
    modelConfiguration = modelBaseDir + "/config/itms-dark-yolov4-tiny-3l-v1.cfg"
    modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov4-tiny-3l-v1_best.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    #

    # Name custom object
    # classes = [" "]
    # Load names of classes by sangkny
    classesFile = modelBaseDir + "/data/itms/itms-classes.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')


    #path for train
    train_path = glob.glob(
        r"" +folder_selected)
    # Images path
    images_path = glob.glob(
        r"" +folder_selected+"\*.jpg")
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for img_path in images_path:
        # Loading image
        print(img_path)
        img = cv2.imread(img_path)
        file_name = img_path.rsplit('\\', 1)[1]
        file_name = file_name.rsplit('.', 1)[0]
        print (file_name)
        img = cv2.resize(img, None, fx=0.7, fy=0.7)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        boxes2 = []
        coordinate_list = []  # for .txt

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    # print("NonArr", class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    a = class_id
                    b = detection[0]
                    c = detection[1]
                    d = detection[2]
                    e = detection[3]
                    print("=====================")
                    print(a,b,c,d,e,confidence)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    boxes2.append([a, b, c, d,e])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    coordinate = class_id, detection[0], detection[1], detection[2], detection[3], "Confidence:", confidence

                    coordinate_for_txt = class_id, detection[0], detection[1], detection[3], detection[4]
                    coordinate_list.append(coordinate_for_txt)
   
                    font = cv2.FONT_HERSHEY_PLAIN
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        filename = os.path.basename(img_path).replace('.jpg', '')
        print("========================All Coordinate=====================")
        # print(coordinate_to_str)
        #coordinate.txt
        with open("result/obj_train_data/%s.txt" %filename, "w") as file: #for .txt
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                    a, b, c, d, e =boxes2[i]
                    ab =  a, b, c, d, e, confidence
                    print("=========================Choosen Coordinate=======================")
                    print(ab)
                    printout = str(a)+" "+ str(b)+" "+ str(c)+" "+ str(d)+" "+ str(e)+ "\n"
                    file.write(printout)

        cv2.imshow('result image', img)
        cv2.waitKey(0)
    #train data
    path = folder_selected
    listdir = os.listdir(path)
    
    with open("result/train.txt", "w") as train:
        for file in listdir:
            train.writelines("data/obj_train_data/"+file+"\n")
            
    print("Arr", indexes)
    print("========================End============================")

    #.names
    with open("result/obj.names", "w") as names:
        names.write(v_class)
    
    #.data
    with open("result/obj.data", "w") as data:
        data_content = "classes = "+ "1" + "\n" + "train = data/train.txt" + "\n" + "names = data/obj.names" + "\n" + "backup = backup/" + "\n"
        data.write(data_content)
    messagebox.showinfo("Notif", "Process Completed")

root = Tk()
root.geometry("800x300")
root.title('SVG Auto Annonate v0.99')

def bar():
    # progress.start(20)
    print("============Browse File=============")
    # print(printed)

def execute():
    main()

s = Style()
# path_frame = Frame(root, bg='blue')
# path_frame.pack(side=TOP)

field_frame = Frame(root)
field_frame.pack(pady=20)

# add_class_btn_frame = Frame(root)
# add_class_btn_frame.pack(pady=10)

btn_group_frame = LabelFrame(root, padx=10, pady=10)
btn_group_frame.pack(pady=20)

execute_info_frame = LabelFrame(root, text="Execution Progress", padx=10, pady=10)
execute_info_frame.pack(pady=20)

exit_btn_frame = Frame(root)
execute_info_frame.pack(side=BOTTOM)

# label_file_explorer = Label(path_frame,
#                             text = "Path . . . .",
#                             width = 70, height = 2,
#                             fg = "blue", bg="snow")

field_label = Label(field_frame, text="Class")
add_class_field = Entry(field_frame)

# add_class_btn = Button(add_class_btn_frame, text="Add Class", command=add_class)

btn_execute = Button(btn_group_frame, text="Browse and Execute", width=25, style='execute_btn.TButton', command=add_class)
# btn_cancel = Button(btn_group_frame, text="Cancel", width=25, style='cancel_btn.TButton')

# progress = Progressbar(execute_info_frame, length=400 ,mode='indeterminate', orient=HORIZONTAL)

# exit_btn = Button(exit_btn_frame, text="Quit", width=30)

# specifying rows and columns
# label_file_explorer.grid(column=0, row=0)

btn_execute.grid(column=0, row=1)

field_label.grid(column=0, row=2, padx=15)
add_class_field.grid(column=1, row=2)

# btn_cancel.grid(column=1, row=0, padx=15)

# progress.grid(column=0, row=0)

#style
s.configure('execute_btn.TButton', background='blue')
# s.configure('cancel_btn.TButton', background='red')


# show_img = cv2.imshow("Deteksi Gambar Balon", img)
# key = cv2.waitKey(0)

root.mainloop()
# cv2.destroyAllWindows()