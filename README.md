# EECS504 Final Project: Human Gesture Recognition Through Skeleton Estimation
## Overview
we proposed a vision-based method that identify human gesture for autonomous vehicles. The method utilizes a human skeleton estimation model to find the locations of the human skeleton key points, and constructs a feature vector to represent the human gesture based on the estimated human skeleton key points. A gesture recognition model is further deployed to identify the gesture.

The human skeleton estimation is following this [CVPR paper](https://arxiv.org/abs/1611.08050).

The original repo can be found [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)


## Pipe-Line
1. Human Skeleton Estimation Model Implementation and Training 
2. Gesture Recognition Model Implementation and Training 

### See the demo video -- demo.mp4

### See the how_to_train_human_skeleton_model.mp4 for how to train the model

## Demo
A demo script is provided to see the whole process (main.ipynp)
In this scrip, a test image is firstly imported, then we use the test image to run the human skeleton estimation model, then construct the feature vector, finally we use the gesture recognition model to interpret the gesture.

To run the main.ipynp, you need to :
1. Install all the packages you need  following the package_spec.txt  (Note that you need to install keras 2.2.0, other version may fail to extract the weights)

2. Install Anaconda and jupyter notebook

2. Download the model weights from this [LINK](https://drive.google.com/open?id=1VJiZfLsHz_VhtQBjlekh6bZIfILsnnzI), put the model_weights.h5 file in the data folder.

3. Run main.ipynp cell by cell



## Human Skeleton Estimation Model Implementation and Training
### Data Set

* Option 1: Download a small sample training set saved by us from this [LINK](https://drive.google.com/open?id=1x_R6ReTPYy3MJ_46kLOIwuNFgeKPZEDy)
* Option 2: Download the COCO data set (65GB) and API following this [REPO](https://github.com/cocodataset/cocoapi)

### Training Procedure
1. Download the training data (6GB) following the [LINK](https://drive.google.com/open?id=1x_R6ReTPYy3MJ_46kLOIwuNFgeKPZEDy)
2. Put the train data in data folder
3. (Optional) Download the pre-trained weights from this [LINK](https://drive.google.com/open?id=1VJiZfLsHz_VhtQBjlekh6bZIfILsnnzI), put the model_weights.h5 file in the data folder
2. cd skeleton_estimation_train
3. python3 train_model_main.py
4. Without pre-trained weights, set RETRAIN = 0

###  Detailed Code
1. train_model_main.py: main code to train the model
2. packages_lib.py: import all the packages we need
3. model_lib.py: all the sub cnn modes we need (vgg, ...) to build the skeleton detection model
4. model_builder.py: use the sub cnns from model_lib.py to build the final model 

### Result
<div align="center">
<img src="sample_test/TestResult/test1_modified.jpg", width="300", height="300">
</div>

##  Gesture Recognition Model Implementation and Training

#### We want to point out that our work is focusing on the Human Skeleton Estimation Model Implementation, and Gesture Recognition is an application that uses the detected human skeleton, which is not our focus in this project.

### Data Set
We can not find a free and public data set that fit out requirement, we manually labeled 1000 images that contain three most common human intentions: pedestrian stopping the vehicle, pedestrian requesting a ride and biker indicating the lane changing, to test the gesture recognition concept. 
1. Install jupyter notebook
2. cd gesture_train
3. run gesture_recognition_train.ipynb

### Result
<div align="center">
<img src="test1_modified_re.jpg", width="300", height="300">
</div>
