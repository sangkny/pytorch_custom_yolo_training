## Training YOLO with Custom Dataset in PyTorch
##### The contents came mostly from the previous owner. More functions for training Yolo have been added.
- `splitTrainAndTest.py` splits the given files into training and testing data as createlist.py can do. 
- In addition, `plotAccLoss.py` can plot the loss information from the training log file by darknet with 
- `./darknet detector train /path/to/xxx.data /path/to/xxx.cfg .path/to/xxx_pretrained_weights_or_intermediate_weights_file > train.log`

##### Test
- To test the model, please use `python object_detect_yolov3.py` after editting the file for a target image

##### Under ubuntu training,
    train.txt val.txt should have only linefeed (\r) ended format in linux unlike LF\CR in windows.
    Absolute path is safe. However, relative path can be used.
##### Under Windows training,
    Again, be careful and check the text file format if it has \lf\cr at the end of each line.

Full story:
https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
