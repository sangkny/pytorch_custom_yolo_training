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

modelBaseDir = "C:/Users/mmc/workspace/yolo"
#modelBaseDir = "C:/Users/SangkeunLee/workspace/yolo"

args.image = "./images/demo.jpg"
args.roiObjects = "./images/demo__roi_iou.txt"

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
with open(args.roiObjects, 'rt') as f:
    gtLabels = f.read().rstrip('\n').split('\n')
    # the is [x y width height objects...] information
    for gtl in gtLabels:
        gtll = gtl.split(' ')  # throw the 0th (class id) element
        btbox = [float(s) for s in gtll[:4]]
        btObj = [int(s) for s in gtll[4:]]
        ROIBoxes.append(btbox)              # ROI info
        ROIObjects.append(btObj)            # Objects in ROI info

items = [i for i in range(0, len(ROIBoxes))]
itemcombs = list(combinations(items, optRoiNum)) # we can access itemcombs[0] =(a, b) itemcombs[0][0] = a

# search around the best combination for optimal rois
gtobjidex = [ i for i in range(0, gtObjectNums)]
bestScore = 0
bestScoreIdx = 0
bestUnion =[]
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
    #tot_score = (r_s1+r_s2+r_area) * r_s3 * (1./max(1, len(s12_inter)))
    tot_score = r_s3
    if(tot_score > bestScore):
        bestScore = tot_score
        bestScoreIdx = itidx
        bestUnion = s12_union
        if(args.showImgDetail):
            debugFrame = frame.copy()
            boxAidx, boxBidx = itemcombs[bestScoreIdx]
            boxA, boxB = ROIBoxes[boxAidx], ROIBoxes[boxBidx]
            drawBox(debugFrame, boxA, '%s' % list(ROIObjects[boxAidx]), color=(255, 0, 255))
            drawBox(debugFrame, boxB, '%s' % list(ROIObjects[boxBidx]), color=(0, 255, 255))
            cv.imshow('Seleted ROIs', debugFrame)
            cv.waitKey(1)
        if(args.debugTextDetail):
            print('>>')

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