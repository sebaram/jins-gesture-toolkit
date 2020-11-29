# jins-gesture-toolkit
This is a gesture input data collecting and training toolkit for Jins MEME. On the data recording applications, first, we can set options, such as sampling rate and measurement range. Then, it collects raw sensor data from the device and saves it to file and stream it through TCP socket communication at the same time. The gesture toolkit is working on the base of this stream collection.

The gesture toolkit contains a traditional classifier pipeline with GUI, which helps non-programmable users to build their gesture classifier. It is written with python using several libraries: flask for main GUI, pygame for data collection prompt, Scikit-learn for model training. The pipeline has four steps: Data collection, Data review, Model training, and Online test. Below is the procedure to use the toolkit.

## Requirements
  - Jins DATA Logger: https://github.com/jins-meme/ES_R-DataLogger-for-Windows/releases 
  - request library: flask, pygame, gTTS, playsound (numpy, pandas, scipy)
  - OS: (fully tested on Windows), macOS(partially tested)

## Instructions

### 1.Gesture Training Part(web)
![Gesture Training Part(web)](https://github.com/sebaram/jins-gesture-toolkit/pic/ocular_mltoolkit_main.JPG)

a.  Type the name of gestures what they want to classify and choose the number of repetition to collect trials and time duration
b. (after data collection) Choose the segmentation method to create ground truth data and review it whether data collected and segmented correctly
c. Choose the preprocessing techniques to apply, choose the model to train, and choose inputs to train the model

### 2.Data Collecting Prompt

![Data Collecting Prompt](https://github.com/sebaram/jins-gesture-toolkit/pic/ocular_mltoolkit_pygame.JPG)
Pree right arrow key in keyboard to continue.



## Acknowledgment
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses)


