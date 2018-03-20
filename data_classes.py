import numpy as np
import h5py as h5
import os
from random import shuffle
from BoxPlotter import BoxPlotter
from yad2k.models.keras_yolo import preprocess_true_boxes

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


def iou(box1, box2):
    # Implement the intersection over union (IoU) between box1 and box2
    #
    # Arguments:
    # box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    # box2 -- second box, list object with coordinates (x1, y1, x2, y2)

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / float(union_area)

    return iou

def count_files(folder):
    """
    Counts the number of files in a folder. Used to count the number of images.
    :param folder: (string) Folder whose contents are to be counted
    :return: (int) number of files in the folder
    """
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])


def get_classes(classes_path):
    """
    Read text file of classes. Text file should have one string per line.
    :param classes_path: path to text file classes
    :return: list of classes
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def l2d(list):
    """
    List to Dictionary. Converts a list to a dictionary where the values are the indices.
    :param list: list
    :return: dictionary
    """
    return {c: i for i, c in enumerate(list)}


def clip_labels(labels):
    """
    Remove zeros at the end of label array, since all images are padded with zeros to have the same number of rows,
    regardless of the number of objects in the image.
    :param labels: ndarray of size (max_boxes, 5)
    :return: ndarray of size (actual_objects, 5)
    """
    objects = ~np.all(labels == 0, axis=-1)
    return labels[objects, ...]


class Yad2kData:
    """
    High-level class for interacting with data and feeding it into yad2k
    Has functions for creating data generators and plotting boxes
    This class relies on the h5 file output by the DataCompiler class
    """
    def __init__(self, h5file_path, extractor=None):
        """
        :param h5file_path: Path to the h5 file output by the DataCompiler class
        :param extractor: Instance of the DataExtractor class for plotting the raw images (optional)
            If this is omitted, the class will simply scale the image data back to the original size using Pillow
        """
        # EDITABLE PARAMS
        self.batch_size = 2
        self.shuffle = False

        # Read info from file
        self.partition = {}
        self.h5file_path = h5file_path  # Path to the h5 file path containing the data compiled by DataCompiler

        # Check if file exists
        if not os.path.exists(self.h5file_path):
            raise FileNotFoundError

        file = h5.File(self.h5file_path, mode='r')

        # Get partition (dictionary of IDs)
        for part_name in file.keys():
            if not part_name == 'info':
                self.partition[part_name] = list(file[part_name]['ids'][:].astype(np.str_))

        # Get classes and anchors
        self.anchors = file['info']['anchors'][:]
        self.classes_list = list(file['info']['classes'][:].astype(np.str_))
        self.classes = l2d(self.classes_list)
        self.image_size = file['info']['image_size'][:]
        self.image_data_size = file['train']['images'].shape[1:3]
        self.file = file  # Leave file open for reading

        # Initialize plotting object
        output_dir = os.path.dirname(h5file_path)
        self.plotter = BoxPlotter(self.image_size, int_to_class=self.classes_list, save_path=os.path.join(output_dir, 'output_images'))
        self.extractor = extractor

        print("From %s imported: %d Test, %d Dev, %d Train Images" % (self.h5file_path, self.m_test(), self.m_dev(), self.m_train()))

    def m_test(self):
        """
        :return: (int) size of test set
        """
        if "test" in self.partition:
            return len(self.partition['test'])
        return 0

    def m_train(self):
        """
        :return: (int) size of training set
        """
        return len(self.partition['train'])

    def m_dev(self):
        """
        :return: (int) size of dev set
        """
        return len(self.partition['dev'])

    def get_num_boxes(self):
        return self.file['train']['labels'].shape[1]

    def get_generator(self, name):
        """
        Return a generator object for extracting data from the h5 file
        :param name: 'train' 'dev' or 'test'
        :return: generator object
        """
        if os.path.exists(self.h5file_path):
            print("Created generators reading from " + self.h5file_path)
            generator = self.__generate(name)
            return generator
        else:
            print("No h5 file found")
            return None

    def __generate(self, name):
        """
        Generates batches of samples
        :param name: 'train' 'dev' or 'test'
        """
        # Infinite loop
        n = len(self.partition[name])
        batches_list = list(range(int(np.floor(float(n) / self.batch_size))))
        group = self.file[name]
        while 1:
            # Generate order of exploration of dataset
            if self.shuffle:
                shuffle(batches_list)

            # Generate batches
            if not batches_list:
                print("No batches!!!")
                yield None
            for j, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, n])  # index of the last image in this batch

                X, y = self.__data_generation(group, i_s, i_e)

                yield X, y

    def __data_generation(self, file, i_s, i_e):
        """
        Extracts data from the h5 file in a consecutive chunk
        :param file: h5 File object (or group)
        :param i_s: start index
        :param i_e: end index
        :return: x, y tuple of training data
            x: list of inputs to yad2k algorithm, [images, labels, masks, boxes, IDs]
               IDs was added to the original algorithm, in order to track data fow
            y: array of zeros, per the yad2k training loss method
        """
        images = file["images"][i_s:i_e, ...]
        labels = file["labels"][i_s:i_e, ...]
        masks = file["detector_masks"][i_s:i_e, ...]
        boxes = file["matching_true_boxes"][i_s:i_e, ...]
        IDs = file["ids"][i_s:i_e, ...].astype(np.str)
        y = np.zeros((self.batch_size, 1))

        return [images, labels, masks, boxes, IDs], y

    def id_to_index(self, id):
        name = None
        index = None
        for part_name, ids in self.partition.items():
            if id in ids:
                name = part_name
                index = ids.index(id)
                break
        return name, index

    def plot_id(self, id):
        """
        Plot the truth boxes for a specific ID. Will check if the ID exists in the h5 file
        :param id: (string) example ID
        """
        # Find which set the id is in, if any
        name, index = self.id_to_index(id)

        # Plot the index if it is found
        if not name is None:
            print("ID found in " + name)
            self.plot_index(name, index)
        else:
            print("ID not found")

    def plot_index(self, name, index):
        """
        Plots truth boxes for specific example
        :param name: 'train' 'dev' or 'test'
        :param index: index of the example to plot, in the specified data set
        """
        # Read the data from the h5 file
        label = self.read_label(name, index)
        image = self.read_image(name, index, orig_size=True)

        # Prep the data for plotting
        label = clip_labels(label)
        box, c = self.label2box(label)

        # Plot the boxes on the image using the BoxPlotter class
        self.plotter.plot_truth(image, box, c)

    def label2box(self, labels):
        """
        Convert labels [x y w h class] to bounding box corners [x1 y1 x2 y2] and class.
        :param label: numpy array of shape (max_boxes, 5) of labels [x y w h class] in decimals
        :return box: numpy array of shape (max_boxes, 4) of boxes [x1 y1 x2 y2] in pixels
        :return class: numpy array of shape (max_boxes, 1) of class integers
        """
        box_xy = labels[:, 0:2]
        box_wh = labels[:, 2:4]
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        boxes = np.hstack([
            box_mins[:, 0:1] * self.image_size[0],   # x_min
            box_mins[:, 1:2] * self.image_size[1],   # y_min
            box_maxes[:, 0:1] * self.image_size[0],  # x_max
            box_maxes[:, 1:2] * self.image_size[1]   # y_max
        ]).astype(np.int)
        classes = labels[:, -1].astype(np.int)
        return boxes, classes

    def read_label(self, name, index):
        """
        Read label data directly from h5 file
        :param name: 'test' 'train' or 'dev'
        :param index: index of the example in the data set
        :return: tuple of image and label data (both ndarrays)
        """
        label = self.file[name]['labels'][index, ...]
        return label

    def read_image(self, name, index, orig_size=True):
        if orig_size:
            # Use extractor to plot the raw image, if it is available
            if not self.extractor is None:
                id = self.partition[name][index]
                return self.extractor.read_image_raw(id)
        return self.file[name]['images'][index, ...]
    
    def compare_prediction(self, id, boxes, classes, scores, plot=False):
        iou_threshold = 0.5
        name, index = self.id_to_index(id)
        
        # Read the data from the h5 file
        label = self.read_label(name, index)

        # Prep the data for plotting
        label = clip_labels(label)
        boxes_true, classes_true = self.label2box(label)

        # Initialization
        num_classes = len(self.classes_list)
        num_true = boxes_true.shape[0]
        num_predictions = boxes.shape[0]

        TP = np.zeros((num_classes,))  # True positive (correct)
        FP = np.zeros((num_classes,))  # False positive (detected but incorrect)
        FN = np.zeros((num_classes,))  # False negative (not detected)
        TD = np.zeros((num_classes,))  # Total detections. Count of true labels

        matches = np.zeros(num_true)
        result = np.zeros(num_predictions)  # track if the box was FP (0) or TP (1)

        # Loop over each predicted detection
        for j, bbhat in enumerate(boxes):
            chat = int(classes[j])

            # Match the bounding box with the bounding box with the greatest IOU
            ious = [iou(bbhat, bb) for bb in boxes_true]
            match_ind = int(np.argmax(ious))  # gives the index of the true label that best matches the prediction
            matches[match_ind] += 1  # count how many times true label is matched

            # Check if it matches the correct class and greater than threshold
            if classes_true[match_ind] == chat and ious[match_ind] > iou_threshold:
                TP[chat] += 1
                result[j] = 1
            else:
                FP[chat] += 1
                result[j] = 0

        # Count all true boxes that were never matched as false negatives
        for i in range(num_true):
            if matches[i] == 0:
                FN[classes_true[i]] += 1

        # Get true counts for stats
        for c in classes_true:
            TD[c] += 1

        if plot:
            image = self.read_image(name, index, orig_size=True)
            y = self.plotter.package_data(boxes_true, classes_true)
            yhat = self.plotter.package_data(boxes, classes, scores, result)
            self.plotter.comparison(y, yhat, id, image)

        return TP, FP, FN, TD
        

class DataExtractor:
    def __init__(self, image_size, image_data_size, output_path='.', anchors=None, classes_list=None):
        """
        Prior to calling this initializer, the subclass MUST declare the following:
            self.classes: A dictionary with class name keys (strings) to integer values
            self.classes_list: List of strings of class names
            self.anchors: List of tuples of anchor boxes
        :param image_path: path to the directory of the pictures
        :param image_size: size of the actual image
        :param image_data_size: size of the image data. Should be square when using YOLO anchors. Must be divisible by 32
        :param output_path: path to the directory to where the data will be written
        """
        self.output_path = output_path
        self.image_size = image_size
        self.image_data_size = image_data_size

        self.print_info = True

        # Stores all the labels in an ndarray of size (m, max_boxes, 5)
        # Labels are in the [x y w h class] format
        self.labels = None

        if anchors is None:
            self.anchors = YOLO_ANCHORS
        else:
            self.anchors = anchors
        if classes_list is None:
            self.classes_list = get_classes('data/model_data/coco_classes.txt')
        else:
            self.classes_list = classes_list
        self.classes = l2d(self.classes_list)

        # Detector masks
        self.m = self.count_examples()
        masks_shape = (self.m, self.grid_size()[1], self.grid_size()[0], len(self.anchors), 1)
        boxes_shape = (self.m, self.grid_size()[1], self.grid_size()[0], len(self.anchors), 5)
        self.detector_masks = np.zeros(masks_shape)
        self.matching_true_boxes = np.zeros(boxes_shape)

    def grid_size(self):
        return self.image_data_size[0] // 32, self.image_data_size[1] // 32

    def max_boxes(self):
        self.read_data()
        return self.labels.shape[1]

    def id_to_index(self, id):
        """
        Convert id (string) to the index it is stored in.
        Used to find the entries in self.detector_masks and self.matching_true_boxes that correspond to an id
        :return: index (int)
        """
        raise NotImplementedError

    def count_examples(self):
        """
        Count the total number of training examples
        :return: number of training examples (int)
        """
        raise NotImplementedError

    def get_IDs(self):
        """
        :return: Return a list of strings of the IDs of the training / dev set
        """
        raise NotImplementedError

    def read_image(self, id):
        """
        :param id: ID of the image to return
        :return: ndarray of (h, w, 3) scaled from 0 to 1
        """
        raise NotImplementedError

    def read_label(self, id):
        """
        Read the label of a single example
        :param id: ID of the label to return
        :return: ndarray of shape (max_boxes, 5). Label should be in [x y w h class] format
        """
        raise NotImplementedError

    def read_data(self):
        """
        Read the label data from the files. This will be very unique to each dataset
        Needs to write self.labels, self.max_boxes
        This function will very likely use self.convert_boxes and self.get_detector_mask (in that order)
        This function should be re-callable without penalty
            (i.e. if data is already read subsequent calls should not do anything)
        """
        raise NotImplementedError

    def get_detector_mask(self, boxes, anchors):
        '''
        Precompute detectors_mask and matching_true_boxes for training.
        Detectors mask is 1 for each spatial position in the final conv layer and
        anchor that should be active for the given boxes and 0 otherwise.
        Matching true boxes gives the regression targets for the ground truth box
        that caused a detector to be active or 0 otherwise.
        Copied from YAD2K retrain_yolo.py
        :param boxes [x y w h class] format
        :param anchors
        :return tuple of detectors_mask and matching true boxes
            detectors_mask: ndarray of shape (m, grid_size_h, grid_size_w, num_anchors, 1)
            matching_true_boxes: ndarray of shape (m, grid_size_h, grid_size_w, num_anchors, 5)
            where grid_size is the size of the final grid
        '''
        detector_save_path = os.path.join(self.output_path, "KITTI-masks.npz")
        if os.path.exists(detector_save_path):
            self.print("Loading detector masks from file...")
            data = np.load(detector_save_path)
            detectors_mask = data['detectors_mask']
            matching_true_boxes = data['matching_true_boxes']
        else:
            self.print("Computing detector masks...")
            detectors_mask = [0 for i in range(len(boxes))]
            matching_true_boxes = [0 for i in range(len(boxes))]
            for i, box in enumerate(boxes):
                detectors_mask[i], matching_true_boxes[i] = \
                    preprocess_true_boxes(box, anchors, [self.image_data_size[1], self.image_data_size[0]])
            detectors_mask = np.array(detectors_mask)
            matching_true_boxes = np.array(matching_true_boxes)
            np.savez(detector_save_path, detectors_mask=detectors_mask, matching_true_boxes=matching_true_boxes)

        return np.array(detectors_mask, dtype=np.bool), np.array(matching_true_boxes)

    def convert_boxes(self, boxes):
        '''
        Converts the boxes to [x, y, w, h, class] in decimals from [class, x1, y1, x2, y2] in pixels for passing into the training algorithm
        :param boxes list of numpy arrays
        :return ndarray of (m, n_b, 5) with n_b equal to the maximum number of boxes (max 24)
                  boxes [x, y, w, h, class] in decimals
        Copied from YAD2K retrain_yolo.py
        '''
         # Box preprocessing.
        orig_size = np.expand_dims(self.image_size, axis=0)
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]  # 397.5, 116.5
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]  # 51, 85
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros((max_boxes - boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(boxes)

    def print(self, string):
        if self.print_info:
            print(string)


class DataCompiler:
    def __init__(self, extractor: DataExtractor, name=None, num_example=None):
        if name is None:
            name = "Yad2k_data.h5"
        self.h5file_path = os.path.join(extractor.output_path, name)

        # User params
        self.test_split = 0.1
        self.dev_split = 0.1

        # Extractor
        self.data = extractor
        IDs = self.data.get_IDs()

        # Set number of training examples
        if num_example is None:
            num_example = self.data.m

        # Split into train and dev
        self.m_dev = int(np.round(num_example * self.dev_split))
        self.m_test = int(np.round(num_example * self.test_split))
        self.m_train = int(num_example - self.m_dev - self.m_test)
        self.m_total = num_example
        test_names = IDs[:self.m_test]
        dev_names = IDs[self.m_test:self.m_dev + self.m_test]
        train_names = IDs[(self.m_dev + self.m_test):self.m_total]
        self.partition = {'train': train_names, 'dev': dev_names, 'test': test_names}

    def create_h5(self):
        file = h5.File(self.h5file_path, mode='w')
        for name, ids in self.partition.items():
            self.__write_to_h5(file, ids, name)
        # Write anchors and classes to "info" group
        anchors = np.array(self.data.anchors)
        classes = np.array(self.data.classes_list).astype(np.string_)
        info_group = file.create_group('info')
        info_group.create_dataset('anchors', anchors.shape, np.float)
        info_group.create_dataset('classes', classes.shape, classes.dtype)
        info_group.create_dataset('image_size', (2,), np.int)
        info_group['anchors'][...] = anchors
        info_group['classes'][...] = classes
        info_group['image_size'][...] = self.data.image_size
        file.close()

    def __write_to_h5(self, file, IDs, name):
        """
        Saves data to h5 file read to read into the generator and into the training function
        Saves the data to named group (i.e. 'train' or 'dev') to split the data
        :param file: h5py dataset to write to
        :param IDs: ID of the images to write
        :param name: name of the group to write to

        Writes the following data:
        images: (m, h, w, 3) ndarray of image data, scaled from 0 to 1, with [w,h] defined by image_data_size
        labels: (m, n_b, 5) ndarray of labels [x, y, w, h, class]
        :return:
        """
        group = file.create_group(name)
        n = len(IDs)
        print("Number of images in " + name + ": " + str(n))
        image_shape = (n, self.data.image_data_size[1], self.data.image_data_size[0], 3)
        label_shape = (n, self.data.max_boxes(), 5)
        masks_shape = self.data.detector_masks.shape
        boxes_shape = self.data.matching_true_boxes.shape

        ID_len = int(len(IDs[0]))
        group.create_dataset("images", image_shape, np.float)
        group.create_dataset("labels", label_shape, np.float32)
        group.create_dataset("ids", (n,), '>S' + str(ID_len))
        group.create_dataset("detector_masks", masks_shape, np.bool)
        group.create_dataset("matching_true_boxes", boxes_shape, np.float32)

        for i, id in enumerate(IDs):
            if (i + 1) % 100 == 0:
                print("Finished %d / %d" % (i + 1, n))
            im = self.data.read_image(id)
            label = self.data.read_label(id)
            index = self.data.id_to_index(id)
            group["images"][i, ...] = im
            group["labels"][i, ...] = label
            group["ids"][i] = np.string_(id)
            group["detector_masks"][i, ...] = self.data.detector_masks[index, ...]
            group["matching_true_boxes"][i, ...] = self.data.matching_true_boxes[index, ...]

