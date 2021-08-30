import tensorflow as tf
saved_model_dir = 'Tensorflow/workspace/models/yolov4-tiny-416'


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('faster_rcnn.tflite', 'wb') as f:
  f.write(tflite_model)