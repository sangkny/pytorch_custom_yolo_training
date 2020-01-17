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
opt.input = "C:/Users/mmc/workspace/yolo/data/itms/itms-train_20200115.log"
opt.out = "C:/Users/mmc/workspace/yolo/data/itms/itms-train_20200115-train-loss-plot.png"

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
    iterations.append(int(lineParts[0].split(':')[0]))
    avg_loss.append(float(lineParts[1].split()[0]))

fig = plt.figure()
for i in range(0, len(lines)):
    plt.plot(iterations[i:i+2], avg_loss[i:i+2], 'r.-')

plt.xlabel('Batch Number')
plt.ylabel('Avg Loss')
if showImgFlag:
    plt.show()
if saveImgFlag:
    fig.savefig(outFile, dpi=1000)

print('Done! Plot saved as {}'.format(outFile))
