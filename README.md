## Training YOLO with Custom Dataset in PyTorch
##### The contents came mostly from the previous owner. More functions for training Yolo have been added.
- `splitTrainAndTest.py` splits the given files into training and testing data as createlist.py can do. 
- `splitTrainAndTest.py` shuffles the data before splitting.
- In addition, `plotAccLoss.py` can plot the loss information from the training log file by darknet with_ 
- `./darknet detector train /path/to/xxx.data /path/to/xxx.cfg .path/to/xxx_pretrained_weights_or_intermediate_weights_file > train.log` for `silent mode`
- `./darknet detector train /path/to/xxx.data /path/to/xxx.cfg .path/to/xxx_pretrained_weights_or_intermediate_weights_file 2>&1 | tee train.log` for `display mode`

##### Test
- To test the model, please use `python object_detect_yolov3.py` after editting the file for a target image
- To test roi-based object, please use `python roi_object_detecton_yolov3.py` with single image 
- To Analyze roi-based object detection, please use `python analysis_roi_object_detection_yolov3.py` with an input and its annotated data.
> > + IOU computation and ROI selection algorithm has been included 

##### Under ubuntu training,
    train.txt val.txt should have only linefeed (\n) ended format in linux unlike \r\n in windows.
    Absolute path is safe. However, relative path can be used.
##### Under Windows training,
    Again, be careful and check the text file format if it has \lf\cr at the end of each line.
    note that only '\r' for mac system
##### Detail Reference 
    "Darknet_Custom_Training_A2Z.txt"** in the working folder

##### **Best Description to conduct train/test Yolov3 step-by-step** 
. Convert2Bo repository   
 
    A.	Annotation converting from xml file        

. Pytorch_custom_yolo_training repository    
    
    A.	splitTrainAndTest.py    
	B.	set the configuration (yolo/config) and data (yolo/data/classes.names and train.txt, test.txt with images/labels folders, please see the /workspace/yolo/)
	C.	In pytorch: train.py for training and converting. However, if you use Docker
		i.	sudo nvidia-docker run –it –v ~/workspace:/workspace –-ipc=host sangkny/darknet:~ /bin/bash
		ii.	inside docker,
			1.	./darknet detector train /path/to/xxx.data /path/to/xxx.cfg .path/to/xxx_pretrained_weights_or_intermediate_weights_file 2>&1 | tee train.log
	D.	plotAccLoss.py 
		i.	to see the Acc/Loss plot
	E.	python object_detect_yolov3.py
		i.	to test the model after editing the file for a target image
	F.  python roi_object_detect_yolov3.py
	    i.  to test roi-based detection results
	    ii. to select region for ROI
	G. python analysis_roi_detection_yolov3.py
	    i.  to select best performance region automatically
	    ii. to inspect some information for debugging

. Yolov3 repository

	A.	It was done with only CPU
	B.	It is fitted for mobile phone and efficient development for yolov3
	C.	It provides a convert tool fro pyTorch to yolov3 and vice versa.



Full story:
https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9