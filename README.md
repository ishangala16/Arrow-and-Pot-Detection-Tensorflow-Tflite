# Arrow and pot detection tensorflow tflite android
Arrow Detection &amp; Pot Detection Project

## Environment Requirements:

* Python 3.7.4
* Tensorflow 2.4.1
* Cuda Toolkit 11.0
* Cudnn 8.0

## Models Used & Tested
* EfficientDet-Lite2
* EfficientNet
* Faster Rcnn
* Ssd_Resnet50_v1
* Ssd_Resnet152_v1
* Yolov4 tiny

## Installation
*  Anaconda Environment
*  Tensorflow 
*  Object Detection Api
* Use **Tensorflow Version: 2.4.1** & **Python Version: 3.7.4** as it works perfectly without any dependancy issues.
* Follow [official installation tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to install all dependancies.

## Collecting Data, Training & Testing 
* Follow the steps given in [Tutorial.ipynb](https://github.com/ishangala16/face-recognition-tensorflow-object-detection-api/blob/main/Tutorial.ipynb).
* Use Tensorflow [Tflite Model Maker](https://www.tensorflow.org/lite/guide/model_maker) to directly create a tflite model.
* Use [Tflite exmaple app](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) for deploying your tflite models 

## Sample Output
* Arrow Detection
![alt-text](https://github.com/ishangala16/arrow-and-pot-detection-tensorflow-tflite-android/blob/main/arrow_detection.png)
* Pot Detection
![alt-text](https://github.com/ishangala16/arrow-and-pot-detection-tensorflow-tflite-android/blob/main/pot_detection.png)
* Arrow Detection in Black & White
![alt-text](https://github.com/ishangala16/arrow-and-pot-detection-tensorflow-tflite-android/blob/main/bw_arrow_detection.png)

