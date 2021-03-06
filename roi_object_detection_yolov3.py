# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 roi object_detection_yolov3.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
from random import randint
import os.path

parser = argparse.ArgumentParser(description='ROI-based Object Detection using YOLOv3 in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--showText', type=int, default=1, help='show text in the ouput.')
parser.add_argument('--ps', type=int, default=1, help='stop each image in the screen.')
parser.add_argument('--showImgDetail', type = int, default = 1, help ='show image in detail')
parser.add_argument('--showImgDetailText', type = int, default= 1, help ='flag to show texts in ROI image')
args = parser.parse_args()

# Initialize the parameters
confThreshold = 0.2 # 0.5  # Confidence threshold
nmsThreshold = 0.2 # 0.4  # Non-maximum suppression threshold

inpWidth = 320# 32*10 #32*10  # 608     #Width of network's input image # 320(32*10)
inpHeight = 320#32*9 #32*9 # 608     #Height of network's input image # 288(32*9) best

modelBaseDir = "C:/Users/mmc/workspace/yolo"
#modelBaseDir = "C:/Users/SangkeunLee/workspace/yolo"
#rgs.image = modelBaseDir + "/data/itms/images/4581_20190902220000_00001501.jpg"
#args.image = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180911_113611_cam_0_bg1x.jpg"
# args.image = "./images/demo.jpg"
args.image = "./images/6085-20200909-191759-1599646679.mp4008310.jpg"
#args.image = "./images/11M-20200602-113827-주간 단독 정지-2.mp4000130.jpg"
#args.image = "./images/20180911_113511_cam_0_000090.jpg"
#args.video = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_192557_cam_0.avi"
#args.video = "E:/Topes_data_related/11M-mpg/11M-20200603-222314-야간 단독 정지 보행.mp4"
args.showText = 0
args.ps = 1
args.showImgDetail = 1
args.showImgDetailText = 1



# Load names of classes
classesFile = modelBaseDir + "/data/itms/itms-classes.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

# modelConfiguration = "/data-ssd/sunita/snowman/darknet-yolov3.cfg";
# modelWeights = "/data-ssd/sunita/snowman/darknet-yolov3_final.weights";
# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3.cfg"
# modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final_20200113.weights"

# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v2.cfg"
# modelWeights = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v2_100000.weights"
# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v3-1.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-tiny_3l-v3-2_86000.weights" # v3-2_86000, v3-3_97000
# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v3-4.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-tiny_3l-v3-4_148000.weights" # v3-2_86000, v3-3_97000 # new anchors
modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v3-2.cfg"
modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-tiny_3l-v3-2_100000.weights" #"C:\Users\mmc\workspace\yolo\data\itms\weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom, color=(0,255,0)):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), color, 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    if args.showText:
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                (0, 255, 255), cv.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
m_startFrame = 0;
outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py_20200113.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    cap.set(cv.CAP_PROP_POS_FRAMES, m_startFrame)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        if(args.ps):
            cv.waitKey(0)
        else:
            cv.waitKey(1)

        break

    bboxes = []
    colors = []
    #bboxes = [(231, 125, 208, 128), (225, 202, 529, 392), (211, 376, 841, 525)]
    # # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # # So we will call this function in a loop till we are done selecting all objects

    print("Press q to quit selecting boxes and start detecting")
    print("Press any other key to select next object")

    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv.selectROI('ROI as many as possible', frame)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        k = cv.waitKey(0) & 0xFF
        #k = cv.waitKeyEx(0) & 0xFF
        print(k)
        if (k == ord("q")):  # q is pressed 113
            break

    subFrames =[]
    print('Selected bounding boxes {}'.format(bboxes))
    for bb in bboxes:
        [bx, by, bwidth, bheight] = bb
        subFrame = frame[by:by+bheight, bx:bx+bwidth]
        subFrame = cv.resize(subFrame, (inpWidth, inpHeight))
        subFrames.append(subFrame)

    # loop for multi-block roi -----------------------
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    # # Create a 4D blob from a frame.
    # blob1 = cv.dnn.blobFromImages(subFrames, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # print("blob1: {}".format(blob1.shape))
    # # Sets the input to the network
    # net.setInput(blob1)
    # # Runs the forward pass to get output of the output layers
    # outs1 = net.forward(getOutputsNames(net))
    # t, _ = net.getPerfProfile()

    classIds = []
    confidences = []
    boxes = []
    etimes = [] # elapse time for net.forward
    for sfidx, sf in enumerate(subFrames):
        # sub frame information
        subclassIds = []
        subconfidences = []
        subboxes = []
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(sf, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        print("subROI: {}, blob: {}".format(sfidx, blob.shape))
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        print(getOutputsNames(net))

        # compute performance time / milisecs
        et, _ = net.getPerfProfile()
        tlabel = et * 1000.0 / cv.getTickFrequency() # milisecs
        etimes.append(tlabel)


        # let's correct coordinates as  
        # corrent only center positions   [x,y, width, height] is  [detection[0], detection[1], detection[2], detection[3]]
        [rcx, rcy, rwidth, rheight] = bboxes[sfidx]
        cnt = 0
        for out in outs:
            print("out.shape : ", out.shape)
            for detection in out:
                # if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                # if scores[classId]>confThreshold:
                confidence = scores[classId]
                # Remove the bounding boxes with low confidence
                if detection[4] > confThreshold:
                    print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                    #print(detection)
                if confidence > confThreshold:
                    center_x = rcx + int(detection[0] * rwidth)
                    center_y = rcy + int(detection[1] * rheight)
                    width = int(detection[2] * rwidth)
                    height = int(detection[3] * rheight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    # sub frame
                    subclassIds.append(classId)
                    subconfidences.append(float(confidence))
                    subboxes.append([left, top, width, height])

                    cnt = cnt + 1
        print('# of candidates for {}-th roi: {}'.format(sfidx, cnt))

        # draw each sub frame information
        if args.showImgDetail:
            subindices = cv.dnn.NMSBoxes(subboxes, subconfidences, confThreshold, nmsThreshold)
            tmpFrame = frame.copy()
            subColor = colors[sfidx]
            cv.rectangle(tmpFrame, (rcx, rcy), (rcx+rwidth, rcy+rheight), subColor, 2)
            #if args.showText:
            textLabel = 'Roi (x,y,width,height, # objs):({}, {}, {}, {}, #{}) in {} msec'.format(rcx, rcy, rwidth, rheight, len(subindices), str(tlabel))
            cv.putText(tmpFrame, textLabel, (rcx, rcy-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            for i in subindices:
                i = i[0]
                box = subboxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                drawPred(tmpFrame, subclassIds[i], subconfidences[i], left, top, left + width, top + height, subColor)
            cv.imshow("subROI:"+str(sfidx), tmpFrame)
            cv.waitKey(1)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print("# of Roi:{}, # of Cands:{}, # of object:{}".format(len(subFrames), len(boxes), len(indices)))
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #t, _ = net.getPerfProfile()
    tot = 0
    for etime in etimes:
        tot = tot + etime

    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    label = 'Inference time: %.2f ms' % (tot)
    print(label)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
    cv.waitKey(1)
