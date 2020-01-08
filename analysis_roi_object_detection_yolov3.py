# Usage example:  python3 analysis_roi_object_detection_yolov3.py --image=bird.jpg --anno bird.txt
# 본 파일은 마우스로 역역을 지정하여 그 영역위주로 object를 윈도우의 크기를 조절하며 가장 좋은 조건을 찾기
# 위한 프로그램임.
# 단순히 특정 지역만을 넣어 나온 결과를 보기 위한 것은
# object_detection_yolov3.py 에 구현되어 있음.
# --------------------------------------------------------------------------------------------------

import cv2 as cv
import argparse
import sys
import numpy as np
from random import randint
import os.path

parser = argparse.ArgumentParser(description='ROI-based Object Detection using YOLOv3 in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--labelImg', help='Path to Label image')
parser.add_argument('--useLabelImg', type=int, default=1, help= 'flag to use label reference')
parser.add_argument('--saveTxt', type = int, default = 1, help='flag to save object detection results')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--showText', type=int, default=1, help='show text in the ouput.')
parser.add_argument('--ps', type=int, default=1, help='stop each image in the screen.')
parser.add_argument('--showImgDetail', type = int, default=1, help ='show image in detail')
parser.add_argument('--showRoiImgDetail', type = int, default=1, help ='show Roi image in detail')
parser.add_argument('--showImgDetailText', type = int, default=1, help ='flag to show texts in ROI image')
parser.add_argument('--analyzeROI', type=int, default= 1, help='flag to trig if roi analysis is conducted or not')
parser.add_argument('--roiMouseInput', type=int, default=0, help='flag for roi mouse input or not')
parser.add_argument('--debugTextDetail', type=int, default=1, help='flag for displaying texts in Detail')

args = parser.parse_args()

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 32*10 #32*10  # 608     #Width of network's input image # 320(32*10)
inpHeight = 32*9 #32*9 # 608     #Height of network's input image # 288(32*9) best

#modelBaseDir = "C:/Users/mmc/workspace/yolo"
modelBaseDir = "C:/Users/SangkeunLee/workspace/yolo"
#rgs.image = modelBaseDir + "/data/itms/images/4581_20190902220000_00001501.jpg"
#args.image = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180911_113611_cam_0_bg1x.jpg"
args.image = "./images/demo.jpg"
args.labelImg = "./images/demo_yolo.txt"
#args.video = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_192557_cam_0.avi"
args.showText = 0
args.ps = 1
args.showImgDetail = 1
args.showRoiImgDetail = 0
args.showImgDetailText = 1
args.debugTextDetail = 0
args.analyzeROI = 1
args.roiMouseInput = 0



# Load names of classes
classesFile = modelBaseDir + "/data/itms/itms-classes.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load annotation information
if not os.path.isfile(args.labelImg):
    print("Label file ", args.labelImg, " doesn't exist")
    sys.exit(1)

# get ground truth labels
GTBoxes = []  # [cx,cy, width, height]
with open(args.labelImg, 'rt') as f:
    gtLabels = f.read().rstrip('\n').split('\n')
    # the is [classid, xcenter, ycenter, width, height] information
    for gtl in gtLabels:
        gtll = gtl.split(' ')[1:]  # throw the 0th (class id) element
        btbox = [float(s) for s in gtll]
        GTBoxes.append(btbox)

# Give the configuration and weight files for the model and load the network using them.

# modelConfiguration = "/data-ssd/sunita/snowman/darknet-yolov3.cfg";
# modelWeights = "/data-ssd/sunita/snowman/darknet-yolov3_final.weights";

modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3.cfg"
modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL_FP16)

def cvtXY2Yolo(size, box): # box is not normalized
    dw = 1. / size[0]  # image width
    dh = 1. / size[1]  # image height
    x = (box[0] + box[1]) / 2.0  # x center
    y = (box[2] + box[3]) / 2.0  # y center
    w = box[1] - box[0]  # object width
    h = box[3] - box[2]  # object height
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))  # xcen, ycen, width, height /image size

def cvtYolo2XY(size, box): # box[cx, cy, width, heigh] is normalized
    dw = size[0]  # image width
    dh = size[1]  # image height
    x = max(0, (box[0] - box[2]/2.0))      # x :left
    y = max(0, (box[1] - box[3]/2.0))      # y : top
    x = x*dw
    y = y*dh
    xx = x + box[2]*dw                    # right
    yy = y + box[3]*dh                    # bottom
    return (round(x), round(y), round(xx), round(yy))  # x, y, xx, yy without normalization


"""
# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
 
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
"""
def bb_union(boxA,boxB):
  x = min(boxA[0], boxB[0])
  y = min(boxA[1], boxB[1])
  w = max(boxA[0]+boxA[2], boxB[0]+boxB[2]) - x
  h = max(boxA[1]+boxA[3], boxB[1]+boxB[3]) - y
  return (x, y, w, h)

def bb_intersection(boxA,boxB):
  x = max(boxA[0], boxB[0])
  y = max(boxA[1], boxB[1])
  w = min(boxA[0]+boxA[2], boxB[0]+boxB[2]) - x
  h = min(boxA[1]+boxA[3], boxB[1]+boxB[3]) - y
  if w<0 or h<0: return (0,0,0,0) #return () # or (0,0,0,0) ?
  return (x, y, w, h)

# IOU implementation
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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

# mouse event
def click_and_crop(event, x, y, flags, param):
    # refPt와 corner selection 변수를 global로 만듭니다.
    global refPt, c_select

    # 왼쪽 마우스가 release 되면 (x, y) 좌표의 갯수에 따라 기록을 시작하거나 끝낸다
    if event == cv.EVENT_LBUTTONUP:
        if len(refPt) < 1: # selection 0
            refPt = [(x, y)]
            c_select = True
        elif len(refPt) <= 3:
            refPt.append((x,y))
            if(len(refPt) == 4):
                c_select = False
                # draw the image
                # print(refPt)

# main function ---------------
# Process inputs
winName = 'Select Best Object Detection Window in OpenCV'
cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)

outputFile = "yolo_out_py.avi"
outputFileTxt = "_roi_iou.txt"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
    outputFileTxt = args.image[:-4] +'__roi_iou.txt'
    outInfoFile = open(outputFileTxt, 'wt')
    ## print (classIndex, xcen, ycen, w, h)
    #outInfoFile.write("%.6f %.6f %.6f %.6f\n" % (xcen, ycen, w, h))

elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
spatialStep = 20 # shift step (stride)
spatialStepX = spatialStep # shift step (stride)
spatialStepY = spatialStep # shift step (stride)
sizeStep = 32
overLapRatio = 0.3         # overlap area ratio 10%

if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while args.analyzeROI > 0:
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

    # create mask with mouse selection
    if args.roiMouseInput:
        image = frame.copy()
        clone = frame.copy()
        # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
        cv.namedWindow("image")
        refPt = []
        cv.setMouseCallback("image", click_and_crop)
        '''
        키보드에서 다음을 입력받아 수행합니다.
        - q : 작업을 끝냅니다.
        - r : 이미지를 초기화 합니다.
        - c : ROI 사각형을 그리고 좌표를 출력합니다.
        '''
        while True:
            # 이미지를 출력하고 key 입력을 기다립니다.
            cv.imshow("image", image)
            key = cv.waitKey(1) & 0xFF

            # 만약 r이 입력되면, crop 할 영열을 리셋합니다.
            if key == ord("r"):
                image = clone.copy()
                refPt=[]

            # 만약 c가 입력되고 ROI 박스가 정확하게 입력되었다면
            # 박스의 좌표를 출력하고 crop한 영역을 출력합니다.
            elif key == ord("c"):
                if len(refPt) == 4:
                    #roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                    roi = cv.polylines(image, [np.array(refPt, np.int32)], True, (0, 255, 0), 2)
                    print(refPt)
                    cv.imshow("ROI", roi)
                    cv.waitKey(0)
            # 만약 q가 입력되면 작업을 끝냅니다.
            elif key == ord("q"):
                break
        # 열린 window를 종료합니다.
        cv.destroyWindow("ROI")
        cv.destroyWindow("image")
    else: # use default ROI Region
        refPt = [(284, 130), (1783, 827), (261, 954), (238, 134)]

    # create the possible ROI with spatial step and Size
    # loop for multi-block roi -----------------------
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    tmask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)    # target mask
    rmask = tmask.copy()                                            # roi mask
    if len(refPt) == 4:
        tmask = cv.fillConvexPoly(tmask, np.array(refPt, np.int32), (255, 255, 255))
        if args.showRoiImgDetail:
            cv.imshow('road mask', tmask)
            cv.waitKey(1)
    else:
        print('no refPt !!!')
    nonZerotmask = cv.countNonZero(tmask)
    curWsizeStep = np.max([2, int(inpWidth / sizeStep) -2])  # current size step, min: 2*sizeStep 64 [-2]
    curHsizeStep = np.max([2, int(inpHeight / sizeStep) -2])  # current size step, min: 2*sizeStep 64 [-2]
    MaxSizeStep = np.min([int(frameWidth/sizeStep), int(frameHeight/sizeStep)])
    bboxes = []
    dstep = (curWsizeStep-curHsizeStep)
    if(curWsizeStep >= curHsizeStep):
        bw = list(range(curWsizeStep,MaxSizeStep))
        bh = list(range(curWsizeStep-dstep,MaxSizeStep-dstep))
    else:
        bw = list(range(curWsizeStep + dstep, MaxSizeStep + dstep))
        bh = list(range(curWsizeStep, MaxSizeStep))

    for wi, w in enumerate(bw):     # width and hight increases as time goes
        brw = w * sizeStep                    # width
        brh = bh[wi] * sizeStep               # height
        for yi in list(range(0, frameHeight, spatialStepY)):
            if yi + brh >= frameHeight:
                continue
            for xi in list(range(0, frameWidth, spatialStepX)):
                if xi + brw >= frameWidth:
                    continue
                # check the condition with area overlap
                # maskimg2 = cv.fillConvexPoly(mask, np.array(
                #     [(bboxes[0][0], bboxes[0][1]), (bboxes[0][0] + bboxes[0][2], bboxes[0][1]),
                #      (bboxes[0][0] + bboxes[0][2], bboxes[0][1] + bboxes[0][3]),
                #      (bboxes[0][0], bboxes[0][1] + bboxes[0][3])], np.int32), (255, 255, 255))
                # reset roi mask to zero
                rmask.fill(0)
                rmask = cv.fillConvexPoly(rmask, np.array(
                    [(xi, yi), (xi + brw, yi),
                     (xi + brw, yi + brh),
                     (xi, yi + brh)], np.int32), (255, 255, 255))
                #inter = np.logical_and(tmask, rmask) # is not working : black all
                inter = cv.bitwise_and(tmask, rmask)
                #area_ratio = float(cv.countNonZero(inter))/nonZerotmask
                area_ratio = float(cv.countNonZero(inter)) / float(brw*brh)
                if(args.showRoiImgDetail):
                    cv.rectangle(inter, (xi, yi), (xi+brw, yi+brh), (255, 0, 255), 2)
                    cv.imshow('intersection', inter)
                    cv.waitKey(1)
                if area_ratio >= overLapRatio:
                    # test if the given ROI contains locations of GT
                    boxcnt = 0
                    for gtbox in GTBoxes:
                        boxA = [xi,yi, xi+brw, yi+brh]
                        boxB = cvtYolo2XY([frameWidth,frameHeight], gtbox)
                        boxI = bb_intersection(boxA, boxB)
                        boxcnt += (
                            1 if boxI[0] == boxB[0] and boxI[1] == boxB[1] and boxI[2] == boxB[2] and boxI[3] == boxB[
                                3] else 0)
                        # cv.rectangle(inter, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (255, 0, 255), 2)
                        # cv.imshow('boxB', inter)
                        # cv.waitKey(1)
                    if boxcnt > int(len(GTBoxes)*0.3): # 1/3  약 5개
                        bboxes.append((xi, yi, brw, brh))

    print('# of candidates : {} regions'.format(len(bboxes)))
    if args.debugTextDetail:
        print(bboxes)

    # loop for multi-block roi -----------------------
    classIds = []
    confidences = []
    boxes = []
    etimes = [] # elapse time for net.forward
    debugFrame = frame.copy()
    #bboxes = bboxes[100:]
    for bidx, bb in enumerate(bboxes):
        [bx, by, bwidth, bheight] = bb
        subFrame = frame[by:by + bheight, bx:bx + bwidth]
        subFrame = cv.resize(subFrame, (inpWidth, inpHeight))
        # sub frame information
        subclassIds = []
        subconfidences = []
        subboxes = []
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(subFrame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        if args.debugTextDetail:
            print("subROI: {}, blob: {}".format(bidx, blob.shape))
            print(getOutputsNames(net))

        # compute performance time / milisecs
        et, _ = net.getPerfProfile()
        tlabel = et * 1000.0 / cv.getTickFrequency() # milisecs
        etimes.append(tlabel)
        # let's correct coordinates as
        # corrent only center positions   [x,y, width, height] is  [detection[0], detection[1], detection[2], detection[3]]
        [rcx, rcy, rwidth, rheight] = bboxes[bidx] # this is bb
        cnt = 0
        # save information
        for out in outs:
            if args.debugTextDetail:
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
        print('# of candidates for {}-th roi: {}'.format(bidx, cnt))

        if args.saveTxt or args.showImgDetail:
            roi_ious = []
        # draw each sub frame information
            subindices = cv.dnn.NMSBoxes(subboxes, subconfidences, confThreshold, nmsThreshold)
            if args.showImgDetail:
                # debugFrame.fill(0)
                debugFrame = frame.copy()
                cv.rectangle(debugFrame, (rcx, rcy), (rcx+rwidth, rcy+rheight), (255, 0, 255), 2)
                #if args.showText:
                textLabel = 'Roi (x,y,width,height, # objs):({}, {}, {}, {}, #{}) in {} msec'.format(rcx, rcy, rwidth,
                                                                                                     rheight,
                                                                                                     len(subindices),
                                                                                                     str(tlabel))
                cv.putText(debugFrame, textLabel, (rcx, rcy-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            for i in subindices:
                i = i[0]
                box = subboxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                # search GT and compute IOU to find out the corresponding objects
                boxB = [left, top, left+width, top+height]
                #roi_ious=[idx if bb_intersection_over_union(boxB, cvtYolo2XY([frameWidth, frameHeight], gtbox)) > 0.5 else '' for idx, gtbox in enumerate(GTBoxes)]
                for idx, gtbox in enumerate(GTBoxes):
                    if bb_intersection_over_union(boxB, cvtYolo2XY([frameWidth, frameHeight], gtbox)) > 0.5:
                        roi_ious.append(idx)

                if args.showImgDetail:
                    drawPred(debugFrame, subclassIds[i], subconfidences[i], left, top, left + width, top + height, (0,255,0))

            if args.showImgDetail:
                # cv.imshow("subROI:"+str(sfidx), tmpFrame)
                cv.imshow("subROI", debugFrame)
                cv.waitKey(1)

            # put the roi_iou_information
            if len(roi_ious) > 0:
                #outInfoFile.write("%.6f %.6f %.6f %.6f\n" % (xcen, ycen, w, h))
                lineText = "%d %d %d %d"%(bx, by, brw, brh) # roi box (x,y, width, height)
                for idx in range(0, len(roi_ious)):
                    lineText = lineText + ' ' + str(roi_ious[idx])
                lineText = lineText + '\n'
                outInfoFile.write(lineText)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print("# of Roi:{}, # of Cands:{}, # of object:{}".format(len(bboxes), len(boxes), len(indices)))
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
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    # close the file
    outInfoFile.close()  # roi iou information
    cv.imshow(winName, frame)
    cv.waitKey(1)
    args.analysisROI = 0
