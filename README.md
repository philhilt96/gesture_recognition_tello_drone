## Hand Gesture Based Navigation of Tello EDU Drone

Isaac Trevenen and Phil Hilt

## Setting Up the Repo

- Program was developed and tested using a 2020 Macbook Air M1
- Python version 3.8.13 was used as the programming language and Pip version 22.0.4 was used as a package manager
- All Python modules used on local machine are located in the "requirements.txt" file, though many may not be essential to the program
**Important Note:** Since this program was developed on an ARM architecture machine a virtualenv was setup in the ENV folder to run dependencies reliant on X86 architecture. To get the tensorflow module to successfully run a wheel was used. This may be a non-issue if the program is run on X86 or X64 architectures.

## Program Structure

- main.py is the entry point of the program and contains the logic for control of the program
- config.txt and args.py contain the configuration of openCV
- calculations.py contains a class to handle FPS calculations along with other needed calculations
- tello.py and stats.py contain the code to make a connection to the drone and send navigation commands
- tello_control.py contains methods for keyboard and gesture testing control of the drone
- gesture_recognition.py contains classes with the methods and logic to collect keypoints via MediaPipe Hands and process into a .csv file with the help of opencv and other python libraries
- the model folder contains the MLP model including:
	- the keypoint.csv file holding the keypoints to train the model
	- the keypoint_classifier.hdf5/.tflite files to specify the MLP model with TensorFlow
- model_train.ipynb is a Jupyter Notebook script to actually train the MLP model using TensorFlow Keras

## General Useage

- After installing all dependencies run main.py while connected to a Tello EDU drone to run the program
- The opencv window of the cameras video feed should open, make sure to click on this window to move keyboard focus to the window
- To close the program at any time hit the ESC key
- To takeoff/land the drone at any time hit the spacebar
- There are three threaded modes to control the drone with the keyboard, gestures or collect keypoints
	- hit 'k' to go into keyboard control where 'a' and 's' can be used to move up/down respectively
	- hit 'g' to go to gesture mode and with the drone in the air hand gestures should control movement of the drone
	- hit 't' to collect keypoints by pressing the number coresponding to the gesture id while making the hand gesture into the drone's camera
		-  make sure to check the gesture id's in keypoint_classifier_labels.csv
		-	this mode is sometimes unstable and will occasionally crash the video feed due to unkown errors, but it should save the keypoints to the .csv even if it crashes