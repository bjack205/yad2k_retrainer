import os
import numpy as np
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from yad2k.models.keras_yolo import yolo_body, yolo_loss, yolo_eval, yolo_head, yolo_boxes_to_corners

import tensorflow as tf


def create_model(image_size, anchors, num_classes, model_file=None, load_pretrained=True, freeze_body=True):
    """
    Generates keras models of YOLOv2. Modified from original version to optionally load the whole model from file, and
    work with any size image.
    :param image_size: [w, h] size of the image data (square when using YOLO anchors)
    :param anchors: list of tuples of anchor boxes
    :param num_classes: number of classes
    :param model_file: path to keras model file. If this is used, the top layer will be maintained,
                       but care must be taken to make sure that the model is compatible with the data
    :param load_pretrained: load the pretrained model (true) or initialize all weights (false). Ignored wth model file.
    :param freeze_body: freeze all weights except for the last layers. Ignored with model file.
    :return model_body: keras model of YOLOv2 with modified output layer, for testing
    :return model: keras model of YOLOv2 with custom loss layer, for training
    Modified version of the original yad2k version in retrain_yolo.py
    """
    if not model_file is None:
        print("USING MODEL FILE")
        model_body = load_model(model_file)
        input_shape = model_body.layers[0].input_shape
        image_input_size = [input_shape[2], input_shape[1]]
        assert np.allclose(image_input_size, image_size)

    detectors_mask_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 1)
    matching_boxes_shape = (image_size[1]//32, image_size[0]//32, len(anchors), 5)

    # Create model input layers.
    image_input = Input(shape=(image_size[1], image_size[0], 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
    ids_input = Input(shape=(1,))  # added input to track IDs

    if model_file is None:
        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), num_classes)

        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if load_pretrained:
            # Save topless yolo:
            topless_yolo_path = os.path.join('data/model_data', 'yolo_topless.h5')
            if not os.path.exists(topless_yolo_path):
                print("CREATING TOPLESS WEIGHTS FILE")
                yolo_path = os.path.join('data/model_data', 'yolo.h5')
                model_body = load_model(yolo_path)
                model_body = Model(model_body.inputs, model_body.layers[-2].output)
                model_body.save_weights(topless_yolo_path)
            topless_yolo.load_weights(topless_yolo_path)

        final_layer = Conv2D(len(anchors)*(5+num_classes), (1, 1), activation='linear')(topless_yolo.output)

        model_body = Model(image_input, final_layer)

    if freeze_body:
        for layer in model_body.layers[:-1]:
            layer.trainable = False

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': num_classes})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input, ids_input], model_loss)

    return model_body, model