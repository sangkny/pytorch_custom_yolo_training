# no import package for independent configuration management

# Initialize the parameters
CONF_THRES = 0.1 #0.5  # Confidence threshold
NMS_THRES  = 0.1 #0.4  # Non-maximum suppression threshold

INPWIDTH  = 32*10  # 608     #Width of network's input image # 320(32*10)
INPHEIGHT = 32*9 # 608     #Height of network's input image # 288(32*9) best

# start video frame number
Video_Start_Frame = (21-4)*60*30+(2-39)*30 # compute the first starting location from a video

# model base dir
ModelBaseDir = "C:/Users/mmc/workspace/yolo"
TEST_IMAGE_PATH ="C:/Users/mmc/workspace/yolo/data/itms/images/demo2.jpg"
TEST_VIDEO_PATH = ""#"E:/Topes_data_related/시나리오 영상/시나리오 영상/20200909PM/6085-20200909-170439-1599638679.mp4"
SHOW_TEXT_FLAG = 1
PS_FLAG = 1

# Load names of classes, please don't include the first directory separator like "/data/..."
CLASSES_FILE = "data/itms/itms-classes.names"

# Give the configuration and weight files for the model and load the network using them.
# Don't include the first directory separator
# -- yolov3 ------
# ------- 3 layers
Model_Configuration = "config/itms-dark-yolov3-tiny_3l-v3-2.cfg"
Model_Weights = "data/itms/weights/itms-dark-yolov3-tiny_3l-v3-2_100000.weights"
# ------- full layers
#Model_Configuration = "config/itms-dark-yolov3-full-2.cfg"
# Model_Weights = "data/itms/weights/itms-dark-yolov3-full-2_100000.weights"
# -- yolov4 -------
# 3l layers
# Model_Configuration = "config/itms-dark-yolov4-tiny-3l-v1.cfg"
# Model_Weights       = "data/itms/weights/itms-dark-yolov4-tiny-3l-v1_best.weights"
# full layers
# Model_Configuration = "config/itms-dark-yolov4-full.cfg"
# Model_Weights       = "data/itms/weights/itms-dark-yolov4-full_92000.weights"

# -----------------------  old test files and configurations   -----------------------------
# Give the configuration and weight files for the model and load the network using them.
# modelConfiguration = "/data-ssd/sunita/snowman/darknet-yolov3.cfg";
# modelWeights = "/data-ssd/sunita/snowman/darknet-yolov3_final.weights";
# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3.cfg"
# modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final_20200113.weights"

# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v3-1.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-tiny_3l-v3-2_100000.weights" #"C:\Users\mmc\workspace\yolo\data\itms\weights"

# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l-v3-2.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-tiny_3l-v3-2_50000.weights" #"C:\Users\mmc\workspace\yolo\data\itms\weights"

# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-full-2.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov3-full-2_100000.weights"
# -- yolov4 -------
# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov4-full.cfg"
# modelWeights = modelBaseDir + "/data/itms/weights/itms-dark-yolov4-full_92000.weights"

# image tests
#args.image = modelBaseDir + "/data/itms/images/20180911_115711_cam_0_001110.jpg" #4581_20190902220000_00001501.jpg"
#args.image = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180911_113611_cam_0_bg1x.jpg"
#args.image = "./images/demo2.jpg"
# video tests
#args.image = "./images/6085-20200909-210107-1599652867.mp4002773.jpg"
#args.video = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_192557_cam_0.avi"
#args.video = "E:/Topes_data_related/11M-mpg/11M-20200603-222314-야간 단독 정지 보행.mp4"
#args.video =  "E:/Topes_data_related/시나리오 영상/시나리오 영상/20200909PM/6085-20200909-170439-1599638679.mp4"