# Usage example:  python3 Pick_Best_POI.py --roiObjects demo_yolo.txt
# 각 ROI 별 object detection 결과에 따라 가장 적정한 roi window를 선택
# --------------------------------------------------------------------------------------------------

import cv2 as cv
import argparse
import sys
import numpy as np
from random import randint
import os.path
from itertools import combinations

parser = argparse.ArgumentParser(description='Best ROI Selection')
parser.add_argument('--image', type = str, help = 'reference image')
parser.add_argument('--roiObjects', type = str, default = './images/demo__roi_iou.txt', help = 'file containing detected objects in ROIs')
parser.add_argument('--saveImg', type = int, default=1, help='flag to save aresult image')
parser.add_argument('--showImgDetail', type = int, default=1, help ='show image in detail')
parser.add_argument('--showRoiImgDetail', type = int, default=1, help ='show Roi image in detail')
parser.add_argument('--showImgDetailText', type = int, default=1, help ='flag to show texts in ROI image')
parser.add_argument('--debugTextDetail', type=int, default=1, help='flag for displaying texts in Detail')

args = parser.parse_args()

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

gtObjectNums = 16       # ground truth object numbers in the given demo file
optRoiNum = 2           # optimal roi numbers

#modelBaseDir = "C:/Users/mmc/workspace/yolo"
modelBaseDir = "C:/Users/SangkeunLee/workspace/yolo"

# args.image = "./images/demo.jpg"
# args.roiObjects = "./images/demo__roi_iou_20200113.txt"
# args.image = "./images/demo_v1.jpg"
# #args.roiObjects = "./images/demo_v1__roi_iou.txt" #demo_v1__roi_iou_20200113
# args.roiObjects = "./images/demo_v1__roi_iou_20200113.txt"

args.image = "./images/20200421_182213-1_0.jpg"
args.roiObjects = "./images/demo4__roi_iou_20200430.txt"


args.saveImg = 1
args.showImgDetail = 1
args.showRoiImgDetail = 0
args.showImgDetailText = 1
args.debugTextDetail = 1

# function definition
def drawBox(frame, boxA, boxAlabel, color=(0,255,0)):
    # Draw a bounding box.
    left, top, right, bottom = int(boxA[0]), int(boxA[1]), int(boxA[0] + boxA[2]), int(boxA[1] + boxA[3])
    cv.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = '(%d, %d, %d, %d) ' % (left, top, int(boxA[2]), int(boxA[3])) + boxAlabel

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    if args.showImgDetailText:
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                (0, 255, 255), cv.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# load detected objects in ROI
if not os.path.isfile(args.roiObjects):
    print("Label file ", args.roiObjects, " doesn't exist")
    sys.exit(1)
# shows ROIs in the image
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)

if(args.showImgDetail):
    cap = cv.VideoCapture(args.image)
    # get frame from the video
    hasFrame, frame = cap.read()
    debugFrame = frame.copy()

# get ground truth labels
ROIBoxes = []  # [x,y, width, height] absolute size
ROIObjects = []
roi_MinMax = [100, 100, 0, 0] # (xmin, ymin), (xmax, ymax) which is (left, top) of each roi
with open(args.roiObjects, 'rt') as f:
    gtLabels = f.read().rstrip('\n').split('\n')
    # the is [x y width height objects...] information
    for gtl in gtLabels:
        gtll = gtl.split(' ')  # throw the 0th (class id) element
        btbox = [float(s) for s in gtll[:4]]
        btObj = [int(s) for s in gtll[4:]]
        ROIBoxes.append(btbox)              # ROI info
        ROIObjects.append(btObj)            # Objects in ROI info
        roi_MinMax[0] = min(roi_MinMax[0], btbox[0])
        roi_MinMax[1] = min(roi_MinMax[1], btbox[1])
        roi_MinMax[2] = max(roi_MinMax[2], btbox[0])
        roi_MinMax[3] = max(roi_MinMax[3], btbox[1])


items = [i for i in range(0, len(ROIBoxes))]
itemcombs = list(combinations(items, optRoiNum)) # we can access itemcombs[0] =(a, b) itemcombs[0][0] = a

# search around the best combination for optimal rois
gtobjidex = [ i for i in range(0, gtObjectNums)]
bestScore = 0
bestScoreIdx = 0
bestUnion =[]
total_combs = len(itemcombs)
processing_step = int(total_combs/10)
best_roi_dist = 0                                # distance between rois
frameHeight = frame.shape[0]
frameWidth = frame.shape[1]
x_dist = (roi_MinMax[0]-roi_MinMax[2]) #*frameWidth
y_dist = (roi_MinMax[1]-roi_MinMax[3]) #*frameHeight
Max_roi_dist = np.sqrt(x_dist*x_dist+y_dist*y_dist)
print(Max_roi_dist)
for itidx, itcombs in enumerate(itemcombs): # (itcombs1, itcombs2)  in itemcombs
    # where itidx is required for roiBoxes locations
    s1 = ROIObjects[itcombs[0]]         # set1
    s2 = ROIObjects[itcombs[1]]         # set2
    s1_box = ROIBoxes[itcombs[0]]       # s1 box
    s2_box = ROIBoxes[itcombs[1]]       # s2 box
    s12_union = list(set(s1).union(s2)) # union
    s12_inter = list(set(s1).intersection(s2)) # intersection
    s4 = list(set(gtobjidex).intersection(s12_union)) # how many matches the results of object detection to GT
    # if len(s12_union) > getObjectNums: it means detected objects are larger than the Ground Truth
    r_s1 = 1./len(s1)                   # smaller s1*s2 is better for separating the ROIs
    r_s2 = 1./len(s2)
    r_s3 = len(s12_union)/gtObjectNums       # bigger the better
    r_area = max(s1_box[2]*s1_box[3], s2_box[2]*s2_box[3])/(s1_box[2]*s1_box[3] + s2_box[2]*s2_box[3])  # separation is better
    roi_dist = np.sqrt((s1_box[0]-s2_box[0])*(s1_box[0]-s2_box[0])+(s1_box[1]-s2_box[1])*(s1_box[1]-s2_box[1]))
    r_dist = roi_dist/Max_roi_dist # the longger the better
    #tot_score = (r_s1+r_s2+r_area) * r_s3 * (1./max(1, len(s12_inter)))
    '''
    # the following score combination according to conditions
    # this combination will find out the all gt and separate two rois as far as possible
    '''
    tot_score = r_s1*0.05 + r_s2*0.05 + r_s3*0.6 + r_dist*0.3

    if(tot_score > bestScore):
        bestScore = tot_score
        bestScoreIdx = itidx
        bestUnion = s12_union
        best_roi_dist = roi_dist
        if(args.showImgDetail):
            debugFrame = frame.copy()
            boxAidx, boxBidx = itemcombs[bestScoreIdx]
            boxA, boxB = ROIBoxes[boxAidx], ROIBoxes[boxBidx]
            drawBox(debugFrame, boxA, '%s' % list(ROIObjects[boxAidx]), color=(255, 0, 255))
            drawBox(debugFrame, boxB, '%s' % list(ROIObjects[boxBidx]), color=(0, 255, 255))
            cv.imshow('Seleted ROIs', debugFrame)
            cv.waitKey(1)
        if(args.debugTextDetail):
            print('Best score: ' + '%.2f' % bestScore
                  + ', boxA:' + '{}'.format(list(ROIObjects[boxAidx]))
                  + ', boxB:' + '{}'.format(list(ROIObjects[boxBidx]))
                  + ', union:' + '{}'.format(bestUnion)
                  + ', roi_dist:' + '%.2f' % (best_roi_dist / Max_roi_dist))
    # processing indicator
    if args.debugTextDetail and (itidx % processing_step == 0):
        print('processing :' + '%.2f' %(itidx/total_combs*100) + '%')

#print(ROIBoxes)
#print(ROIObjects)
print("best score:{}, idx:{}, union:{}".format(bestScore, bestScoreIdx, bestUnion))

debugFrame = frame.copy()
boxAidx, boxBidx = itemcombs[bestScoreIdx]
boxA, boxB = ROIBoxes[boxAidx], ROIBoxes[boxBidx]
drawBox(debugFrame, boxA, '%s' % list(ROIObjects[boxAidx]), color=(255, 0, 0))
drawBox(debugFrame, boxB, '%s' % list(ROIObjects[boxBidx]))
cv.imshow('Seleted ROIs', debugFrame)
cv.waitKey(0)
if args.saveImg and args.image :
    saveImgFile = args.image[:-4] + "_bestRoi_20200113.jpg"
    cv.imwrite(saveImgFile, debugFrame)
    print('Result has been save: {}'.format(saveImgFile))