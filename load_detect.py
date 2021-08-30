WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
# CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
# CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CONFIG_PATH = MODEL_PATH+'/faster_rcnn/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/faster_rcnn/'

import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
import math
import serial,time


# arduino = serial.Serial('COM7', 115200, timeout=.1)
# time.sleep(1)

len_arrow_pixel = 2267
# focal = 606 #focal lenght of Hp Omen 
focal = 1647 #focal length of pocof1 in cm (pixel to cm converted)

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-5')).expect_partial() #change checkpoint according to model

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True: 
    angle_in_degrees = 0
    dist = 0    
    ret, frame = cap.read()
    # frame = cv2.imread("C:/Users/Ishan/Downloads/Arrow Images/Arrow Test images/IMG_20210607_095245.jpg")

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_orange = np.array([5,116,190], dtype=np.uint8)
    # upper_orange = np.array([13,255,255], dtype=np.uint8)
    # lower_white = np.array([0,0,99], dtype=np.uint8)
    # upper_white = np.array([179,35,255], dtype=np.uint8)
    
    # # Threshold the HSV image to get only white colors
    # mask = cv2.inRange(hsv, lower_white, upper_white)
    # print(np.shape(hsv))
    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(blackAndWhiteImage, last_axis)
    training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')

   # print("train",np.shape(training_image))

    # converted = tf.image.rgb_to_grayscale(frame)
    # # back2 = tf.make_ndarray(converted)
    # print("converted",np.shape(converted))

    # blackAndWhiteImage = np.resize(blackAndWhiteImage,(blackAndWhiteImage.shape[0], blackAndWhiteImage.shape[1], 3))
    # # blackAndWhiteImage = blackAndWhiteImage.reshape((blackAndWhiteImage.shape[0], blackAndWhiteImage.shape[1], -1))
    # print("sdadadad", np.shape(blackAndWhiteImage))
    # blackAndWhiteImage[0,0,0] = frame[0,0,0]
    # blackAndWhiteImage[0,0,1] = frame[0,0,1]
    # blackAndWhiteImage[0,0,2] = frame[0,0,2]

    # cv2.imshow('game', image_np)
    # 
    # rgb = tf.image.grayscale_to_rgb(gray)
    # print("is of type", type(rgb))
    # print (np.expand_dims(image_np, 0))
    # tester = np.expand_dims(blackAndWhiteImage, 2)
    # print( gray[0,0,0])
    # print( gray[0,0,1])
    # print( gray[0,0,2])
    
    # print("qeqweasd")
    # print(frame[0,0,0])
    image_np = np.array(frame)
    # image_np = np.array(training_image) #for black and white
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) # np.expand_dims(image_np, 0) --> image_np_expanded
    # print("benstokes !!",type(input_tensor))
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    


    # print("box",detections['detection_boxes'])

    coords = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                np.squeeze(detections['detection_boxes']),
                np.squeeze(detections['detection_classes']+label_id_offset).astype(np.int32),
                np.squeeze(detections['detection_scores']),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2,
                min_score_thresh=0.7,
                agnostic_mode=False)
    
    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # print("dipzet",boxes)
    # print(coords)
    
    #getting bounding box coordinates
    detection_boxes = detections['detection_boxes']
    detection_classes = detections['detection_classes'].astype(int) + label_id_offset
    # detection_scores = detections['detection_scores']

    # Scale to pixel co-o rdinates
    # print("dsdsd",image_np_with_detections.shape)
    
    
    #detection_boxes[0] *= 480
    #detection_boxes[1] *= 640

    

    detection_boxes[0][0] *= 480
    detection_boxes[0][2] *= 480
    detection_boxes[0][1] *= 640
    detection_boxes[0][3] *= 640

    detection_boxes[1][0] *= 480
    detection_boxes[1][2] *= 480
    detection_boxes[1][1] *= 640
    detection_boxes[1][3] *= 640

    # print(detections['detection_classes'][0],detection_boxes[0])
    # print(detection_classes[1] ,detection_boxes[1])


    xheadcen = (int)((detection_boxes[0][1] + detection_boxes[0][3])/2)
    yheadcen = (int)((detection_boxes[0][0] + detection_boxes[0][2])/2)
    # print("avg",detections['detection_classes'][0],xheadcen)

    xtailcen = (int)((detection_boxes[1][1] + detection_boxes[1][3])/2)
    ytailcen = (int)((detection_boxes[1][0] + detection_boxes[1][2])/2)
    # print("avg",detections['detection_classes'][1],xtailcen)

    if (xheadcen - xtailcen)> 0 or (xheadcen - xtailcen)< 0:
        m = ((yheadcen - ytailcen)/(xheadcen - xtailcen))
        len= ((((xheadcen - xtailcen )**2) + ((yheadcen-ytailcen)**2) )**0.5)
        dist=int((len*focal)/len_arrow_pixel)
        angle_in_radians = math.atan(m)
        angle_in_degrees = round(math.degrees(angle_in_radians))
    else:
        angle_in_degrees = 0

    # data = arduino.readline()
    # if data == '':
        # arduino.write(angle_in_degrees)

    d1 = str(angle_in_degrees)


    print("Angle is ",angle_in_degrees)
    # print("Distance is ",dist)
    # arduino.write(d1.encode())
    # time.sleep(1)
    

    #print(detection_boxes[1] , detection_boxes[1][0],detection_boxes[1][1])
    #print(round(detection_boxes[0][0]) , round(detection_boxes[0][1]) , round(detection_boxes[1][0] , round(detection_boxes[1][1])))
    # img = cv2.resize(frame,(800,600))
    image_np_with_detections = cv2.circle(image_np_with_detections ,(detection_boxes[0][1],detection_boxes[0][0]), radius = 10 ,color = (255,0,0) , thickness = 5 )
    image_np_with_detections = cv2.circle(image_np_with_detections ,(detection_boxes[1][1],detection_boxes[1][0]), radius = 10 ,color = (255,0,0) , thickness = 5 )

    # image_np_with_detections = cv2.line(image_np_with_detections, (detection_boxes[0][3],detection_boxes[0][0]), (detection_boxes[1][1],detection_boxes[1][2]), (255,0,0), 5)
    image_np_with_detections = cv2.line(image_np_with_detections, (xheadcen,yheadcen), (xtailcen,ytailcen), (255,255,0), 2)
    
    # print('Rounded head', xheadcen , yheadcen)
    # print('Rounded tail', xtailcen , ytailcen)
    # print('Head',detection_boxes[0])
    # print('Tail',detection_boxes[1])



    # Select person boxes
    # cond = (detection_classes == 'Head') & (detection_scores >= 0.7)
    # person_boxes = detection_boxes[cond, :]
    # person_boxes = np.round(person_boxes).astype(int)
    # print('arrow coords',np.array(person_boxes))
    # print("is of type", type(blackAndWhiteImage))

    # print(person_boxes)
    
    #cv2.imshow('res', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('BW',  blackAndWhiteImage)
    # cv2.imshow('Train', training_image)
    image_np_with_detections = cv2.putText(image_np_with_detections, "Angle = " + str(angle_in_degrees), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    # image_np_with_detections = cv2.putText(image_np_with_detections, "Distance = " + str(dist), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (640, 480)))
    # cv2.imwrite('img.jpg' , image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
