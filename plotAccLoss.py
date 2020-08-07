#
# usage: python plotAccLoss.py --input /path/to/logfile from darknet.

import sys
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/data/darknet.log", help="training log plot")
parser.add_argument("--showImgFlag", type=int, default=1, help="flag to show output result")
parser.add_argument("--saveImgFlag", type=int, default=1, help="flag for output results")
parser.add_argument("--out", type=str, default="./data/training_loss_plot.png", help="output graphic name")


opt = parser.parse_args()

# parameter settings
# opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-v3-full-HighGpu.log" #
# opt.out = "./logplots/itms-train-v3-20200518-full-train-loss-plot.png"

# --------- 20200729 new data set for yojoo road labelling
# opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-v3-3l-highGPU-20200729.log" #"itms-train-full-1-highGPU-20200731.log"
# opt.out = "./logplots/itms-train-v3-3l-highGPU-20200729-train-loss-plot.png"
# opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-full-1-highGPU-20200731.log" #" full version "
# opt.out = "./logplots/itms-train-full-1-highGPU-20200731-train-loss-plot.png"
# -------- 20200731 randominzed data set
# opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-v3-3l-highGPU-20200729_random.log" #" randomized version "
# opt.out = "./logplots/itms-train-v3-3l-highGPU-20200729_random-train-loss-plot.png"
# # --------- 20200804 scratch version for yejoo road labelling dataset
# opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-v3-3l-highGPU-20200804_scratch.log" # tiny - 3l
# opt.out = "./logplots/itms-train-v3-3l-highGPU-202008004_scratch-train-loss-plot.png"
# --------- 20200806 ne anchor (xx-v3-4.cfg) version for yejoo road labelling dataset
opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train-v3-3l-highGPU-20200806_tl_150000.log" # tiny - 3l
opt.out = "./logplots/itms-train-v3-3l-highGPU-202008006_new_anchors-train-loss-plot.png"

logFile = opt.input;
showImgFlag = opt.showImgFlag
saveImgFlag = opt.saveImgFlag
outFile = opt.out;

plot_item = "avg" # "avg"

lines = []
for line in open(logFile): # sys.argv[1]
    if "avg" in line:
        lines.append(line)

iterations = []
avg_loss = []

print('Retrieving data and plotting training loss graph...')
for i in range(len(lines)):
    lineParts = lines[i].split(',')
    if ":" in lineParts[0]:
        try:
            iterations.append(int(lineParts[0].split(':')[0]))
        except:
            print('read Error at line# {} : {}'.format(i, lineParts))
            continue
        avg_loss.append(float(lineParts[1].split()[0]))

fig = plt.figure()
#for i in range(0, len(lines)):
#    plt.plot(iterations[i:i+2], avg_loss[i:i+2], 'r.-')
plt.plot(iterations, avg_loss, 'r.-')

plt.xlabel('Batch Number')
plt.ylabel('Avg Loss')
if showImgFlag:
    plt.show()
if saveImgFlag:
    fig.savefig(outFile, dpi=1000)

print('Done! Plot saved as {}'.format(outFile))
#compute min location
import numpy as np
idx = np.argmin(avg_loss)
loss = avg_loss[idx]
iter_number = iterations[idx]
print('minimum loss value is {} at {} iteration'.format(loss, iter_number))
