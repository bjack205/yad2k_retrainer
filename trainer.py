import os
# uncomment to turn off GPU (if it doesn't fit in memory)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import numpy as np
from prettytable import PrettyTable
from create_model import create_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from data_classes import Yad2kData, DataExtractor, clip_labels
from yad2k.models.keras_yolo import yolo_head, yolo_eval
import keras.backend as K


def filter_classes(class_ints, class_list, class_filt):
    """
    Returns a mask for the class_ints that are in the class_filt.
    Used to only look at classes you care about
    :param class_ints: list of class integers
    :param class_list: list of strings. Converts class_ints to string names
    :param class_filt: list of string names that you want to keep
    :return: boolean nparray of the classes you want to keep
    """
    class_names = [class_list[int(c)] for c in class_ints]
    filter = [name in class_filt for name in class_names]
    return np.array(filter)


COCO_CLASSES = {'person', 'bicycle', 'car', 'train', 'truck'}


class Trainer:
    def __init__(self, h5file_path):
        # Set if you want to load a model from file (doesn't reconstruct the weights of the top layer)
        self.model_file = None

        # Weights to be loaded by the model before training
        self.weights_file = None

        # Data
        self.data = Yad2kData(h5file_path)
        self.output_path = os.path.dirname(h5file_path)

        # Flags
        self.model_loaded = False

        self.classes_filt = COCO_CLASSES

    def get_model(self, freeze_body=True):
        if self.model_loaded:
            print("WARNING: Model Already loaded! Make sure you aren't doubling up the model!")
        model_body, model = create_model(self.data.image_data_size,
                                         self.data.anchors,
                                         len(self.data.classes),
                                         model_file=self.model_file,
                                         freeze_body=freeze_body)
        self.model_loaded = True
        return model_body, model

    def retrain(self, run_name, num_epochs, fine_tune=True, batch_size=None):
        # Callback functions
        logging = TensorBoard()
        checkpoint_tuning = ModelCheckpoint(run_name + "_checkpoint.h5", monitor='val_loss',
                                            save_weights_only=True, save_best_only=True)

        # Get the model, freezing the body
        model_body, model = self.get_model(freeze_body=fine_tune)

        # Fine Tuning
        model_body.load_weights(self.weights_file)
        model.load_weights(self.weights_file)

        # Compile the Keras model
        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        # Get the data generators
        if not batch_size is None:
            self.data.batch_size = batch_size
        else:
            batch_size = self.data.batch_size
        train_gen = self.data.get_generator('train')
        dev_gen = self.data.get_generator('dev')

        # Train the model
        model.fit_generator(generator=train_gen,
                            steps_per_epoch=self.data.m_train() // batch_size,
                            validation_data=dev_gen,
                            validation_steps=self.data.m_dev() // batch_size,
                            callbacks=[logging, checkpoint_tuning],
                            epochs=num_epochs
                            )

        # Save the model
        print("Finished Fine-tuning, saving weight as " + run_name)
        model.save_weights(run_name + "_.h5")

    def test(self, part_name):
        model_body, _ = self.get_model(freeze_body=False)
        model_body.load_weights(self.weights_file)
        self.test_model(model_body, part_name)

    def test_model(self, model_body, part_name):
        score_threshold = 0.5
        iou_threshold = 0.5
        part_name = 'test'

        # Evaluate the output of the model body
        yolo_outputs = yolo_head(model_body.output, self.data.anchors, len(self.data.classes))
        input_image_shape = K.placeholder(shape=(2,))
        max_boxes = self.data.get_num_boxes()
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            input_image_shape,
            max_boxes=max_boxes,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold)

        # Get tensorflow session
        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        # Initialize variables to store predictions and true labels
        m = len(self.data.partition[part_name])
        IDs = []
        boxes_pred = []
        scores_pred = []
        classes_pred = []

        # Get the generator
        gen = self.data.get_generator(part_name)

        tic = time.time()
        # Loop over each image and test
        for i in range(m):
            # Get next training sample
            x, _ = next(gen)

            # Extract information from sample
            image = x[0]  # decimal values from 0 to 1
            ID = x[-1][0]
            IDs.append(ID)

            # Run Prediction
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model_body.input: image,
                    input_image_shape: [self.data.image_size[1], self.data.image_size[0]],
                    K.learning_phase(): 0
                })

            # Process outputs
            out_boxes = out_boxes[:, [1, 0, 3, 2]]  # outputs boxes [y2, x2, y1, x2]
            # print(str(out_boxes.shape[0]) + " objects detected")
            self.data.compare_prediction(ID, out_boxes, out_classes, out_scores, 1)

            # Append predictions and pad to be the same size as true labels
            boxes_pred.append(out_boxes)
            scores_pred.append(out_scores)
            classes_pred.append(out_classes)

            if (i + 1) % 25 == 0:
                print("Finished Predictions for %d / %d" % (i + 1, m))
        toc = time.time() - tic
        print(toc)
        print("Predictions per second: %.2f" % (m / toc))

        # Convert to numpy arrays
        IDs = np.array(IDs)

        np.savez(os.path.join(self.output_path, "Predictions.npz"), boxes=boxes_pred, scores=scores_pred, classes=classes_pred, ids=IDs)
        self.calc_metrics(IDs, boxes_pred, classes_pred, scores_pred)

    def calc_metrics(self, ids=None, boxes=None, classes=None, scores=None):

        if ids is None and boxes is None and classes is None and scores is None:
            preds = np.load(os.path.join(self.output_path, "Predictions.npz"))
            ids = preds['ids']
            boxes = preds['boxes']
            scores = preds['scores']
            classes = preds['classes']

        # Parameters
        iou_threshold = 0.5  # Threshold for an accurate localization
        plot = True  # Plot the images with bounding boxes

        # Important vars
        num_classes = len(self.data.classes)

        # Set up variables
        m = len(boxes)
        TP = np.zeros((num_classes,))  # True positive (correct)
        FP = np.zeros((num_classes,))  # False positive (detected but incorrect)
        FN = np.zeros((num_classes,))  # False negative (not detected)
        TD = np.zeros((num_classes,))  # Total detections. Count of true labels

        # Loop over all of the images
        for i in range(m):
            ID = ids[i]
            tp, fp, fn, td = self.data.compare_prediction(ID, boxes[i], classes[i], scores[i], plot=plot)
            TP += tp
            FP += fp
            FN += fn
            TD += td

        # Calculate precision and recall
        eps = 1e-9
        mAP = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        total_detections = TP + FP

        mask = [name in self.classes_filt for name in self.data.classes_list]
        header = np.array(self.data.classes_list)[mask]
        header = np.hstack(('0', header))
        T = PrettyTable(list(header))
        T.add_row(np.hstack(("TP", TP[mask])))
        T.add_row(np.hstack(("FP", FP[mask])))
        T.add_row(np.hstack(("FN", FN[mask])))
        T.add_row(np.hstack(("TD", total_detections[mask])))
        T.add_row(np.hstack(("TL", TD[mask])))
        T.add_row(np.hstack(("mAP", np.round(mAP[mask], 2))))
        T.add_row(np.hstack(("RCL", np.round(recall[mask], 2))))
        print(T)


if __name__ == "__main__":
    h5_path = "data/new/KITTI_1000.h5"
    trainer = Trainer(h5_path)
    trainer.weights_file = 'coco_fine_tuning_best.h5'
    trainer.retrain('retrain', num_epochs=10, fine_tune=False, batch_size=1)
    trainer.test(part_name='test')