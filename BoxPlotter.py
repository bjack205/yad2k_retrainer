from PIL import Image, ImageDraw, ImageColor, ImageFont
import os
import numpy as np

class BoxPlotter:
    """
    Object to plot bounding boxes for predicted and true labels
    """
    def __init__(self, image_size, int_to_class, save_path=None):
        """
        :param image_folder: folder containing the images
        :param image_size: size of the resized images
        """
        # Set label font
        self.font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image_size[1] + 0.5).astype('int32'))
        self.image_size = image_size

        # Set bounding box colors
        self.color_pred = [ImageColor.getrgb('red'),  # False Positive
                      ImageColor.getrgb('green')]  # True Positive
        self.color_true = [ImageColor.getrgb('orange'),  # False Negative
                      ImageColor.getrgb('blue')]  # Detected true box
        self.save_path = save_path
        self.int_to_class = int_to_class

    def package_data(self, boxes, classes, scores=None, results=None):
        y = {"boxes": boxes, "classes": classes}
        if not scores is None:
            y["scores"] = scores
        if not results is None:
            y["results"] = results
        return y

    def comparison(self, y, yhat, id, image_data):
        """
        Plots both predicted and true bounding boxes for comparison
        :param y: dictionary of a true label data. One entry of the "y" input to the CalcMetrics function
        :param yhat: dictionary of predicted label data
        :return: Nothing. Displays a plot to the screen
        """
        # Open image and set up drawing variables
        image_data = image_data / np.max(image_data) * 255
        image_data = image_data.astype(np.uint8)
        # plt.imshow(image_data)
        # plt.show()

        image = Image.fromarray(image_data)
        image = image.resize(self.image_size, Image.BICUBIC)
        draw = ImageDraw.Draw(image)

        # Plot the boxes
        self.truth_boxes(draw, y['boxes'], y['classes'])
        self.prediction_boxes(draw, yhat['boxes'], yhat['classes'], yhat['scores'])
        # image.show()

        if not self.save_path is None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            image.save(os.path.join(os.path.join(self.save_path, id + ".png")))

        # Cleanup
        del draw

    def prediction_boxes(self, draw, boxes, classes, scores):
        """
        Plot bounding boxes for predicted labels. Box label includes class name and confidence
        :param draw: PIL.ImageDraw.draw object
        :param yhat: dictionary of predicted detection data
        :return: Nothing
        """
        # Loop over each object detected in the image
        num_detections = scores.shape[0]
        for j in range(num_detections):
            # Extract out import info from dictionary
            classname = self.int_to_class[classes[j]]
            score = scores[j]
            result = 1  # int(yhat['result'][j])
            box = boxes[j, :]

            # Set label and color
            label = '{} {:.2f}'.format(classname, score)
            color = self.color_pred[result]

            # Plot the boxes
            self.plot_box(draw, box, label, color)

    def plot_truth(self, image_data, boxes, classes):
        image_data = image_data / np.max(image_data) * 255
        image_data = image_data.astype(np.uint8)

        image = Image.fromarray(image_data)
        image = image.resize(self.image_size, Image.BICUBIC)
        draw = ImageDraw.Draw(image)
        self.truth_boxes(draw, boxes, classes)
        image.show()

    def truth_boxes(self, draw, boxes, classes):
        """
        Plot bounding boxes for true labels. Box label includes class name and (Truth)
        :param draw: PIL.ImageDraw.draw object
        :param boxes: numpy array of shape (num_objects, 4) of boxes [x1 y1 x2 y2] in pixels
        :param classes: numpy array of shape (num_objects, 1) of class integers
        :return: Nothing
        """
        # Loop over each object detected in the image
        num_true = classes.shape[0]
        for j in range(num_true):
            # Extract important info from dictionary
            classname = self.int_to_class[classes[j]]
            result = 1
            box = boxes[j, :]

            # Set label and color
            label = '{} {}'.format(classname, "(Truth)")
            color = self.color_true[result]

            # Plot the boxes
            self.plot_box(draw, box, label, color)

    def plot_box(self, draw, box, label, color):
        """
        Actual routine for plotting the bounding boxes and labels on a PIL images
        :param draw: PIL.ImageDraw.draw object
        :param box: numpy array [x1, y1, x2, y2]
        :param label: string to include in the label
        :param color: color of the bounding box and label
        :return: Nothing
        """
        label_size = draw.textsize(label, self.font)

        # Draw bounding box
        draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                       outline=color)
        # Draw label
        if box[1] - label_size[1] >= 0:
            text_origin = np.array([box[0], box[1] - label_size[1]])
        else:
            text_origin = np.array([box[0], box[1] + 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)