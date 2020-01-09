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

# load detected objects in ROI
if not os.path.isfile(args.roiObjects):
    print("Label file ", args.roiObjects, " doesn't exist")
    sys.exit(1)

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
for itidx, itcombs in enumerate(itemcombs): # (itcombs1, itcombs2)  in itemcombs
    # where itidx is required for roiBoxes locations
    s1 = ROIObjects[itcombs[0]] # set1
    s2 = ROIObjects[itcombs[1]] # set2
    s12_union = list(set(s1).union(s2)) # uniton
    s4 = list(set(gtobjidex).intersection(s12_union)) # how many matches the results of object detection to GT






print(ROIBoxes)
print(ROIObjects)